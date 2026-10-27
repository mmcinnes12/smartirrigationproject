# Cleaned agricultural field datasets

Processed **10 existing downloaded datasets**, comprising **84 separate series**, **3,998,338 hourly rows** and **1,903,282 candidate forecasting windows**. No extra datasets were fetched for this processing pass. Five datasets are distinct ICOS sites from one network, not five independent data collection networks.

Every compressed CSV contains site, country/region, crop, crop provenance, soil context, irrigation method, sensor depths where available, and time basis. Metadata is also supplied in `dataset_metadata.json`. Unknown metadata remains unknown/null. Original downloaded files remain in the workspace's work folder.

## Suitability matrix

| Dataset | Series | Hourly rows | Candidate windows | Main use / limitation |
|---|---:|---:|---:|---|
| gebesee | 1 | 52,608 | 40,158 | Weather + moisture; annual crops and physical sensor depths unknown. |
| lonzee | 1 | 84,720 | 64,804 | Weather + moisture; annual crop chronology/depths unknown; interim product. |
| lamasquere | 1 | 40,920 | 30,873 | Irrigated rotation context, weather + moisture; no event history or annual crop labels. |
| klingenberg | 1 | 192,864 | 111,434 | Long weather/moisture record; substantial sensor gaps; depths unknown. |
| selhausen | 1 | 61,368 | 55,448 | Good weather/moisture coverage; annual crops/depths unknown. |
| cook | 42 | 3,372,768 | 1,505,955 | Strong crop/soil metadata and multiple depths; dryland; no weather joined. |
| flanders | 19 | 111,180 | 25,153 | Best vegetable crop metadata; 15 cm replicates; logger temperature only, event history unavailable. |
| gallipoli | 1 | 14,597 | 11,168 | Moisture, partial weather and observed irrigation flow; olive and moisture-unit assumptions need confirmation. |
| virginia | 16 | 58,529 | 49,603 | Corn/cotton trial; 2020 crop identities unknown; weather units unverified and retained separately. |
| berambadi | 1 | 8,784 | 8,686 | 2024 moisture, soil temperatures and rain; crop and management unknown. |

## Recommended starting points

- **Cook and Flanders** for models conditioned on documented crop and soil metadata. Cook supplies annual crop codes and measured texture; Flanders supplies planting-based crop episodes and measured retention data.
- **Selhausen, Gebesee and Lonzée** for weather-driven moisture forecasts, while treating annual crop labels and sensor depths as unknown.
- **Virginia** for comparing documented 2021 corn/cotton irrigation treatments. Treatment names are not irrigation event histories.
- **Gallipoli** for exploratory irrigation-flow features, conditional on the stated unit/crop assumptions. Do not treat it as a fully verified production training source.

## Processing and use

Timestamps are regularised to hourly bins. Moisture/temperature/weather use means; verified interval rain uses sums. Each value requires a full nominal hour of samples (1 hourly, 2 half-hourly or 4 quarter-hourly; 12 five-minute rain samples). No gaps are interpolated. Duplicate timestamps are averaged. Documented missing codes and out-of-range values become null. Percentage VWC is converted to m³/m³ where documented. Soil moisture remains depth-specific; channel 1 is the primary target except Flanders, which uses the median of at least two same-depth replicates.

Use rows with `window_24h_valid=true`: they have 24 consecutive valid primary-moisture hours and an observed target 24 hours later. Suspect 48-hour exact flatlines and freezing primary soil temperatures are excluded from these windows. This is a conservative heuristic, not a full sensor calibration audit. Other predictors can still be missing. Known crop-episode boundaries are excluded; unknown chronology cannot be checked. Labels based on planting dates do not establish harvest dates. Hourly means and timestamps are defined in `training_contract.json`.

Retain source time conventions; do not combine these files as if all timestamps were UTC. Forecast targets and retrospective quality flags must not become input features. Split by site and time, with a 48-hour boundary embargo, before learning preprocessing. Candidate counts overlap heavily and are not independent sample counts.

Field capacity, wilting point and root depth were not invented. Flanders pF observations retain their measured names. ICOS sensor indices are not physical depths. Unverified Virginia weather units and Gallipoli wind units remain separately named source fields. Gallipoli flow was not converted to invented irrigation duration or volume. Missing irrigation records are null; zeros are used only for documented rainfed/non-irrigated management.

## Existing unsuitable downloads kept out of the training package

Apple workbook: daily observations cannot supply measured subdaily history. SEnviro: raw moisture sensor counts lack a verified VWC calibration. Strawberry: reported sensor percentage lacks verified VWC calibration. Iowa orchard requested SoilVue columns: downloaded values were empty. Iowa AHTI4: suspect weather values require further sensor validation. These were not forced into the common VWC target.

## Files and verification

`cleaned_datasets/` contains one gzip-compressed CSV per series. `schema_mappings.json` records original headers, conversions and aggregation. `quality_audit.json` records source dimensions, timing, gaps, invalid values and coverage. `suitability_matrix.csv` gives aggregate counts and source links. `validation.json` confirms every exported file was read back and checked for hourly, sorted, unique timestamps, embedded metadata, exact 24-hour target alignment and valid-window requirements. The ZIP contains all these deliverables.

Sources are attached per dataset in the metadata and matrix; observed measurements, researched site context, explicit assumptions and unknown values are distinguished. These are cleaned forecasting candidates, not evidence that all ten have complete crop metadata or irrigation histories.
