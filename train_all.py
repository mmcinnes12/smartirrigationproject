"""10-dataset 24-hour-ahead moisture forecasting: Persistence, RF, LSTM.
Run: python train_all.py --epochs 15 --max-samples 30000
Uses the provided original ZIP directly. See README for methods and caveats.
"""
import argparse
import io
import json
import os
import random
import re
import zipfile
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
ZIP = BASE / 'data' / 'processed_agricultural_datasets.zip'
OUTPUT = BASE / 'results'
HISTORY = 24
HORIZON = 24
EMBARGO = 48
# Use only actual observed readings or contemporaneously-known metadata. No precomputed future targets.
DYNAMIC = [
    'soil_moisture_m3m3',
    *[f'soil_moisture_{i}_m3m3' for i in range(1, 7)],
    *[f'soil_temperature_{i}_c' for i in range(1, 7)],
    'air_temperature_c','relative_humidity_pct','rain_mm',
    'pressure_kpa','wind_speed_ms','radiation_wm2',
    'irrigation_duration_s','irrigation_volume_l',
    'irrigation_flow_lps_observed_mean',
]
STATIC_NUMERIC = [
    'field_capacity_m3m3','wilting_point_m3m3','root_depth_cm',
    *[f'soil_depth_{i}_cm' for i in range(1, 7)],
    *[f'soil_{kind}_{i}_pct' for kind in ['sand','silt','clay'] for i in range(1, 6)],
    *[f'bulk_density_{i}_gcm3' for i in range(1, 6)],
]
STATIC_CATEGORICAL = ['series_id','crop','crop_code','soil_type','irrigation_method']


def tf_setup(seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf


def read_series(z, member):
    with z.open(member) as f:
        df = pd.read_csv(f, compression='gzip', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df


def get_split_frames(df):
    """Chronological 70/15/15 by series; windows stay in one partition.
    Strict +/-48-hour gap at *each* boundary, plus target within partition.
    These windows have end timestamp t and label timestamp t+24.
    """
    n = len(df)
    if n < 300:
        return {}
    b1, b2 = int(n * .70), int(n * .85)
    valid = df['window_24h_valid'].fillna(False).astype(str).str.lower().isin(['true','1','1.0'])
    valid_idx = np.flatnonzero(valid.to_numpy())
    valid_idx = valid_idx[(valid_idx >= HISTORY - 1) & (valid_idx + HORIZON < n)]
    # Verify that each 24-hour history and future target span is hourly.
    ts = df['timestamp'].to_numpy(dtype='datetime64[ns]')
    step = np.diff(ts) == np.timedelta64(1,'h')
    bad = np.r_[0, np.cumsum(~step)]
    good = bad[valid_idx + HORIZON] == bad[valid_idx - (HISTORY-1)]
    valid_idx = valid_idx[good]
    return {
        'train': valid_idx[(valid_idx + HORIZON < b1 - EMBARGO)],
        'val': valid_idx[(valid_idx - (HISTORY-1) >= b1 + EMBARGO) & (valid_idx + HORIZON < b2 - EMBARGO)],
        'test': valid_idx[(valid_idx - (HISTORY-1) >= b2 + EMBARGO)],
    }


def prepare_file(df, split_indices, rng, max_per_series):
    for feature in DYNAMIC + STATIC_NUMERIC:
        if feature not in df:
            df[feature] = np.nan
    for feature in STATIC_CATEGORICAL:
        if feature not in df:
            df[feature] = 'unknown'
    arr = df[DYNAMIC].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    # Categorical values are contemporaneous to forecast origin, not future target.
    num = df[STATIC_NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    cats = df[STATIC_CATEGORICAL].fillna('unknown').astype(str).to_numpy(dtype=object)
    clock = df['timestamp']
    # Features known at t: local timestamp hour/day-of-year (source time bases not aligned).
    hour = clock.dt.hour.to_numpy()
    doy = clock.dt.dayofyear.to_numpy()
    time_features = np.stack([
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    ], axis=1).astype(np.float32)
    target = pd.to_numeric(df['target_soil_moisture_24h_m3m3'],errors='coerce').to_numpy(dtype=np.float32)
    res = {}
    for split, indices in split_indices.items():
        indices = indices[np.isfinite(target[indices]) & np.isfinite(arr[indices,0])]
        if len(indices)>max_per_series:
            indices=np.sort(rng.choice(indices,size=max_per_series,replace=False))
        if len(indices)==0: continue
        windows = indices[:,None] - np.arange(HISTORY-1,-1,-1)[None,:]
        X = np.concatenate([arr[windows], time_features[windows]],axis=2)
        res[split] = dict(x=X, static=num[indices], cats=cats[indices],
                          y=target[indices], persistence=arr[indices,0],
                          timestamp=df['timestamp'].iloc[indices].astype(str).to_numpy(),
                          series=cats[indices,0])
    return res


def merge(parts, split):
    selected=[x[split] for x in parts if split in x]
    if not selected:return None
    return {key:np.concatenate([p[key] for p in selected],axis=0) for key in selected[0]}


def subsample(data, maximum, seed):
    if len(data['y']) <= maximum:return data
    ids=np.sort(np.random.default_rng(seed).choice(len(data['y']),maximum,replace=False))
    return {k:v[ids] for k,v in data.items()}


def engineer(train,val,test):
    # Training-only medians, means and standard deviations. Masks explicitly encode missingness.
    d=train['x'].shape[-1]
    flat=train['x'].reshape(-1,d)
    with np.errstate(all='ignore'):
        med=np.nanmedian(flat,axis=0)
    med=np.where(np.isfinite(med),med,0).astype(np.float32)
    filled=np.where(np.isfinite(flat),flat,med)
    mu=filled.mean(axis=0); sd=np.maximum(filled.std(axis=0),1e-5)
    s=train['static'].shape[-1]
    with np.errstate(all='ignore'):
        smed=np.nanmedian(train['static'],axis=0)
    smed=np.where(np.isfinite(smed),smed,0).astype(np.float32)
    st=np.where(np.isfinite(train['static']),train['static'],smed)
    smu=st.mean(axis=0); ssd=np.maximum(st.std(axis=0),1e-5)
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    enc.fit(train['cats'])
    def transform(part):
        z=part['x']; mask=np.isfinite(z).astype(np.float32)
        z=(np.where(np.isfinite(z),z,med)-mu)/sd
        sx=part['static']; smask=np.isfinite(sx).astype(np.float32)
        sx=(np.where(np.isfinite(sx),sx,smed)-smu)/ssd
        cat=enc.transform(part['cats'])
        # Static metadata duplicated over the 24 historical steps for LSTM.
        static=np.concatenate([sx,smask,cat],axis=1).astype(np.float32)
        seq=np.concatenate([z.astype(np.float32),mask,
                             np.broadcast_to(static[:,None,:],(len(z),HISTORY,static.shape[1]))],axis=2)
        return seq
    outputs=[transform(v) for v in [train,val,test]]
    params=dict(med=med,mu=mu,sd=sd,static_median=smed,static_mean=smu,
                static_sd=ssd,encoder=enc,dynamic=DYNAMIC,static_numeric=STATIC_NUMERIC,
                static_categorical=STATIC_CATEGORICAL)
    return outputs,params


def metrics(y, pred):
    return dict(n=len(y),mae=float(mean_absolute_error(y,pred)),
                rmse=float(np.sqrt(mean_squared_error(y,pred))),
                r2=float(r2_score(y,pred)) if len(y)>1 else None,
                bias=float(np.mean(pred-y)))


def make_plots(name, y, preds, root):
    k=min(400,len(y)); fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(np.arange(k),y[:k],label='Observed',lw=1.7)
    for model,pred in preds.items(): ax.plot(np.arange(k),pred[:k],label=model,alpha=.8,lw=1)
    ax.set(xlabel='Test samples (chronological within sampled series)',ylabel='Soil moisture (m³/m³)',
           title=f'{name}: 24h-ahead test predictions (first {k} samples)')
    ax.legend();fig.tight_layout();fig.savefig(root/'prediction_comparison.png',dpi=150);plt.close(fig)


def run_one(z, dataset, files, args, tf):
    out=OUTPUT/dataset;out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    parts=[]
    # Per-series cap avoids letting extremely long series dominate and controls RAM.
    cap=max(120, int(np.ceil(args.max_samples/max(len(files),1)*1.8)))
    for filename in files:
        df=read_series(z,filename)
        indices=get_split_frames(df)
        part=prepare_file(df,indices,rng,cap)
        if part:parts.append(part)
        del df
    data={key:merge(parts,key) for key in ('train','val','test')}
    if any(v is None or len(v['y'])<40 for v in data.values()):
        print(f'SKIP {dataset}: insufficient train/validation/test samples')
        return []
    # Cap independently within each split; chronological split assignment is unchanged.
    limits=dict(train=args.max_samples,val=max(300,args.max_samples//5),
                test=max(300,args.max_samples//5))
    for i,split in enumerate(data):data[split]=subsample(data[split],limits[split],args.seed+i)
    train,val,test=(data[k] for k in ('train','val','test'))
    (xtr,xval,xte),params=engineer(train,val,test)
    ytr,yval,yte=(v['y'] for v in (train,val,test))
    joblib.dump(params,out/'preprocessing.joblib')
    predictions={'Persistence':test['persistence'].astype(np.float32)}
    print(f'{dataset}: train={len(ytr)}, val={len(yval)}, test={len(yte)}, features={xtr.shape[-1]}',flush=True)
    rf=RandomForestRegressor(n_estimators=args.trees,max_depth=args.rf_depth,
                             min_samples_leaf=3,n_jobs=args.jobs,random_state=args.seed)
    rf.fit(xtr.reshape(len(xtr),-1),ytr)
    predictions['RandomForest']=rf.predict(xte.reshape(len(xte),-1))
    joblib.dump(rf,out/'random_forest.joblib')
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(HISTORY,xtr.shape[-1])),
        tf.keras.layers.LSTM(args.lstm_units,dropout=.1),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),loss='mse',metrics=['mae'])
    stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)
    history=model.fit(xtr,ytr,validation_data=(xval,yval),epochs=args.epochs,
                      batch_size=args.batch_size,callbacks=[stop],verbose=2,shuffle=True)
    predictions['LSTM']=model.predict(xte,batch_size=args.batch_size,verbose=0).ravel()
    model.save(out/'lstm.keras')
    pd.DataFrame(history.history).to_csv(out/'lstm_training_history.csv',index=False)
    pd.DataFrame({'timestamp_at_t':test['timestamp'],'series_id':test['series'],
                  'actual_t_plus_24h':yte,**{f'pred_{key}':val for key,val in predictions.items()}}).to_csv(out/'test_predictions.csv',index=False)
    make_plots(dataset,yte,predictions,out)
    results=[]
    for key,pred in predictions.items():
        row=dict(dataset=dataset,model=key,**metrics(yte,pred))
        results.append(row)
        print(f'  {key:<15} MAE={row["mae"]:.5f} RMSE={row["rmse"]:.5f} R2={row["r2"]:.3f}',flush=True)
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    (out/'settings.json').write_text(json.dumps({'series':len(files),'sample_counts':{k:len(v['y']) for k,v in data.items()},
                                                  'seed':args.seed,'history':HISTORY,'forecast_h':HORIZON,
                                                  'embargo_h':EMBARGO},indent=2))
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',type=Path,default=ZIP)
    parser.add_argument('--datasets',nargs='*',default=None,help='Subset e.g. berambadi selhausen')
    parser.add_argument('--max-samples',type=int,default=30000,help='Max train rows per dataset')
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--trees',type=int,default=150)
    parser.add_argument('--rf-depth',type=int,default=18)
    parser.add_argument('--lstm-units',type=int,default=48)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--jobs',type=int,default=-1)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if not args.zip.exists():parser.error(f'Missing dataset ZIP: {args.zip}')
    random.seed(args.seed);np.random.seed(args.seed)
    tf=tf_setup(args.seed)
    OUTPUT.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(args.zip) as z:
        files={}
        for n in z.namelist():
            m=re.fullmatch(r'cleaned_datasets/([^/]+)/[^/]+\.csv\.gz',n)
            if m:files.setdefault(m.group(1),[]).append(n)
        if args.datasets:
            unknown=set(args.datasets)-set(files)
            if unknown:parser.error('Unknown datasets: '+', '.join(sorted(unknown)))
            files={k:v for k,v in files.items() if k in args.datasets}
        print('Dataset groups:',', '.join(files),flush=True)
        all_results=[]
        for name,members in files.items():
            try:
                all_results.extend(run_one(z,name,members,args,tf))
            except Exception as e:
                print(f'FAILED {name}: {type(e).__name__}: {e}',flush=True)
                if args.datasets:raise
            finally:
                tf.keras.backend.clear_session()
        if all_results:
            leaderboard=pd.DataFrame(all_results)
            leaderboard.to_csv(OUTPUT/'all_model_metrics.csv',index=False)
            pivot=leaderboard.pivot(index='dataset',columns='model',values='rmse')
            pivot.to_csv(OUTPUT/'rmse_comparison.csv')
            print('\nRMSE comparison (lower is better):\n',pivot.to_string())
            print(f'\nSaved files to: {OUTPUT}')
        else:print('No datasets completed successfully.')

if __name__=='__main__':main()
