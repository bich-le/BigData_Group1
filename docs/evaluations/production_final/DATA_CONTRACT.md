# Production input contract

## Inspected sources

| Source | Classification | Use in this Production verification |
|---|---|---|
| `archive (2).zip/flights.csv` | Raw BTS flight events; 31 columns including categorical route fields, scheduled fields, eligibility flags and `ARRIVAL_DELAY` | Usable for fitting the full raw-input pipelines. |
| `archive (2).zip/airports.csv`, `airlines.csv` | Raw lookup CSVs | Not required by the baseline full pipeline. |
| `data/model-ready/{train,test,base}` | Processed Parquet feature tables | Legacy-model-only; not used to fit or score raw full pipelines. |
| `../OOT_Data.zip` | Processed OOT Parquet (`oot_data`, `X_oot`, `y_oot`) | Legacy-model-only unless raw columns are independently reconstructed. |

## Raw full-pipeline serving event

The saved Production pipelines accept a JSON flight event containing:

`AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `MONTH`, `DAY`,
`DAY_OF_WEEK`, `SCHEDULED_TIME`, `SCHEDULED_DEPARTURE`, `DISTANCE`,
`CANCELLED`, `DIVERTED`, plus optional `event_id`.

`ARRIVAL_DELAY` is required only when fitting/evaluating, never for serving.
The pipeline derives `scheduled_hour`, imputes numerics, indexes/encodes the
three categoricals, scales, applies PCA and then scores the RF.

## Compatibility boundary

The legacy artifacts (`random_forest_*_spark`) require the historical ordered
model-ready feature vector. The new full artifacts reject that contract and
serve raw events instead. No claim of raw OOT scoring is made because the
available OOT/model-ready data is processed rather than raw-schema compatible.
