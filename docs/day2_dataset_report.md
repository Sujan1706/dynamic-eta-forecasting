# Day 2 Phase 1: Synthetic ETA Training Dataset Quality & Inspection Report

> [!NOTE]
> **Synthetic Dataset Provenance & Disclaimer**:
> This dataset was generated entirely via the Day 1 discrete-event synthetic train-running simulator and Day 1 feature engineering pipeline (`backend.features.feature_builder.FeatureBuilder`). All timetables, disruptions, and delays are synthetic for MVP machine learning development and **DO NOT** represent real historical Indian Railways operational logs.

---

## 1. Executive Summary

A comprehensive quality audit was conducted on the generated supervised learning training dataset located at [`data/processed/synthetic_train_eta_dataset.csv`](file:///d:/Projects/dynamic-eta-forecasting/data/processed/synthetic_train_eta_dataset.csv) and accompanied by metadata at [`data/processed/dataset_metadata.json`](file:///d:/Projects/dynamic-eta-forecasting/data/processed/dataset_metadata.json).

The dataset satisfies all requirements outlined in the PRD and Day 2 specification:
- **Zero missing values** across all columns ($100\%$ completeness).
- **Physical validity verified**: zero negative distances, zero negative or zero remaining times, zero negative delays, and zero physically impossible speeds ($> 200\text{ km/h}$).
- **Strong ground-truth correlation**: target `actual_remaining_minutes` correlates strongly with `distance_to_go_km` ($r = 0.9837$) and `scheduled_time_to_go_min` ($r = 0.9633$).
- **Realistic disruption distribution**: $30.37\%$ of samples capture operational disruptions across 5 standard railway event types.

---

## 2. Dataset Core Metrics & Dimensions

| Inspection Metric | Value | Details / Scope |
| :--- | :--- | :--- |
| **Total Rows (Samples)** | **6,125** | Observation checkpoints sampled across running segments |
| **Total Journeys** | **200** | Simulated multi-day journeys (25 synthetic dates $\times$ 8 trains) |
| **Unique Trains** | **8** | Indian Railways express & superfast passenger trains |
| **Unique Routes** | **3** | Eastern Trunk, Western Trunk, and Southern Corridor |
| **Origin Checkpoints (Stations)** | **17** | Stations where train state and features were sampled |
| **Destination / Target Stations** | **18** | Next upcoming target stations for ETA prediction |
| **Unique Route Segments** | **18** | Distinct inter-station railway segments |
| **Total Features & Columns** | **20** | 9 identifier/metadata fields + 11 PRD feature inputs |

### Unique Trains in Dataset
1. **12028** - Shatabdi Express (KSR Bengaluru $\to$ MGR Chennai Central)
2. **12260** - Duronto Express (New Delhi $\to$ Howrah)
3. **12302** - Howrah Rajdhani Express (New Delhi $\to$ Howrah)
4. **12306** - Kolkata Rajdhani Express via Patna (New Delhi $\to$ Howrah)
5. **12610** - Chennai Express (KSR Bengaluru $\to$ MGR Chennai Central)
6. **12926** - Paschim Express (New Delhi $\to$ Mumbai Central)
7. **12952** - Mumbai Rajdhani Express (New Delhi $\to$ Mumbai Central)
8. **12954** - August Kranti Tejas Rajdhani (New Delhi $\to$ Mumbai Central)

### Unique Routes in Dataset
- **Route 1**: New Delhi (`NDLS`) $\to$ Howrah (`HWH`) via `CNB`, `PRYJ`, `DDU`, `GAYA`, `DHN`, `ASN` (Eastern Trunk, 1,447 km)
- **Route 2**: New Delhi (`NDLS`) $\to$ Mumbai Central (`MMCT`) via `MTJ`, `KOTA`, `RTM`, `BRC`, `ST` (Western Trunk, 1,386 km)
- **Route 3**: KSR Bengaluru (`SBC`) $\to$ MGR Chennai Central (`MAS`) via `BNC`, `KJM`, `BWT`, `KPD` (Southern Corridor, 359 km)

### Checkpoints (Stations Sampled)
- **Origins**: `SBC`, `BNC`, `KJM`, `BWT`, `KPD`, `NDLS`, `CNB`, `PRYJ`, `DDU`, `GAYA`, `DHN`, `ASN`, `MTJ`, `KOTA`, `RTM`, `BRC`, `ST` (17 stations)
- **Next Target Halts**: `BNC`, `KJM`, `BWT`, `KPD`, `MAS`, `CNB`, `PRYJ`, `DDU`, `GAYA`, `DHN`, `ASN`, `HWH`, `MTJ`, `KOTA`, `RTM`, `BRC`, `ST`, `MMCT` (18 stations)

---

## 3. Data Completeness & Missing Values

Every column has 0 missing, null, or NaN values:

| Column Name | Data Type | Null Count | Missing % | Status |
| :--- | :--- | :--- | :--- | :--- |
| `journey_id` | String (`object`) | 0 | 0.00% | PASS |
| `train_id` | Integer (`int64`) | 0 | 0.00% | PASS |
| `route_id` | Integer (`int64`) | 0 | 0.00% | PASS |
| `station` | String (`object`) | 0 | 0.00% | PASS |
| `target_station` | String (`object`) | 0 | 0.00% | PASS |
| `timestamp` | ISO Datetime (`object`) | 0 | 0.00% | PASS |
| `distance_to_go_km` | Float (`float64`) | 0 | 0.00% | PASS |
| `scheduled_time_to_go_min` | Float (`float64`) | 0 | 0.00% | PASS |
| `num_intermediate_halts` | Integer (`int64`) | 0 | 0.00% | PASS |
| `current_delay_min` | Float (`float64`) | 0 | 0.00% | PASS |
| `delay_trend_3pt` | Float (`float64`) | 0 | 0.00% | PASS |
| `active_speed_restriction_flag` | Integer (`int64`) | 0 | 0.00% | PASS |
| `congestion_score_downstream` | Float (`float64`) | 0 | 0.00% | PASS |
| `hist_avg_delay_this_section` | Float (`float64`) | 0 | 0.00% | PASS |
| `hist_recovery_rate_section` | Float (`float64`) | 0 | 0.00% | PASS |
| `weather_flag` | Integer (`int64`) | 0 | 0.00% | PASS |
| `day_type` | Integer (`int64`) | 0 | 0.00% | PASS |
| `actual_remaining_minutes` *(Target)* | Float (`float64`) | 0 | 0.00% | PASS |
| `disruption_present` | Integer (`int64`) | 0 | 0.00% | PASS |
| `is_synthetic` | Boolean (`bool`) | 0 | 0.00% | PASS |

---

## 4. Target Variable Analysis (`actual_remaining_minutes`)

The supervised target is the true elapsed time in minutes from the observation checkpoint until the train reaches the target station stop line:

$$\text{actual\_remaining\_minutes} = \frac{T_{\text{arrival}} - T_{\text{observation}}}{60\text{ seconds}}$$

### Summary Statistics

| Statistic | Value (Minutes) | Interpretation |
| :--- | :--- | :--- |
| **Minimum** | **1.00 min** | Final approach checkpoint ($\ge 80\%$ segment progress) |
| **25th Percentile ($Q_1$)** | **27.00 min** | Short inter-station hops and late segment progress |
| **Median ($Q_2$)** | **57.00 min** | Typical mid-segment remaining travel time |
| **Mean ($\mu$)** | **68.28 min** | Average travel duration to the next station stop |
| **75th Percentile ($Q_3$)** | **97.00 min** | Early checkpoint on longer trunk sections |
| **95th Percentile** | **176.00 min** | Long trunk sections with accumulated delays |
| **Maximum** | **313.00 min** | Maximum delayed run on longest section (e.g., `MTJ` $\to$ `KOTA`, 324 km) |
| **Standard Deviation ($\sigma$)** | **52.93 min** | Realistic variance across varied inter-station distances |

### Target Correlations with Key Features
- **Correlation with `distance_to_go_km`**: $+0.9837$ (strongly linear, physically realistic)
- **Correlation with `scheduled_time_to_go_min`**: $+0.9633$ (tight alignment with baseline timetable)
- **Correlation with `current_delay_min`**: $-0.1295$ (independent section dynamics; long sections had lower starting delays, short sections had varied delays)

---

## 5. Disruption & Operational Event Distributions

### Overall Disruption Ratio
- **Normal Operations (`disruption_present == 0`)**: **4,265 samples (69.63%)**
- **Disrupted Operations (`disruption_present == 1`)**: **1,860 samples (30.37%)**

### Distribution of Day Types
The simulation spanned a 25-day operational window covering regular weekdays, weekends, and gazetted holidays:

| Day Type Code | Classification | Sample Count | Percentage |
| :---: | :--- | :---: | :---: |
| `0` | Weekday (Monday–Friday) | 4,165 | 68.00% |
| `1` | Weekend (Saturday–Sunday) | 980 | 16.00% |
| `2` | Gazetted Holiday (Synthetic schedule) | 980 | 16.00% |
| **Total** | | **6,125** | **100.00%** |

### Event Types & Active Feature Flags
The simulator injects 5 realistic event types stochastically during segment transit:
1. `SIGNAL_HALT`: Unplanned signal stop (6–22 min stationary delay).
2. `CONGESTION`: Downstream traffic bottleneck (speed reduction factor 0.35–0.65).
3. `SPEED_RESTRICTION`: Engineering track work or caution order (30, 45, or 50 km/h ceiling).
4. `UNSCHEDULED_HALT`: Technical brake inspection or unscheduled stop (5–18 min stationary delay).
5. `WEATHER`: Fog or torrential rain caution (speed reduction factor 0.5–0.7, 50 km/h cap).

#### Active Event Snapshot Flags at Checkpoint Time
When an observation is sampled while an event is actively in effect, the corresponding feature reflects the condition:

| Feature / Indicator | Condition | Sample Count | Sample % | Operational Effect |
| :--- | :---: | :---: | :---: | :--- |
| `active_speed_restriction_flag` | $= 1$ | 46 | 0.75% | Max speed restricted to 30–50 km/h |
| `congestion_score_downstream` | $> 0$ | 47 | 0.77% | Traffic dampening score (max 0.65) |
| `weather_flag` | $= 1$ | 88 | 1.44% | Adverse weather / fog caution active |
| `current_delay_min` | $> 0$ | 4,124 | 67.33% | Cumulative running delay (mean 16.21 min, max 108.91 min) |
| `delay_trend_3pt` | $> 0$ | 2,752 | 44.93% | Positive rate of delay accumulation |

> [!NOTE]
> Transient stationary events like `SIGNAL_HALT` and `UNSCHEDULED_HALT` directly increment `current_delay_min` and manifest in downstream timetable knock-ons even after the train resumes cruising.

---

## 6. Physical Validity & Anomaly Audit

A rigorous verification for potential numerical corruption or non-physical artifacts was conducted:

| Check | Criterion | Found | Status |
| :--- | :--- | :---: | :---: |
| **Negative Distance** | $\text{distance\_to\_go\_km} < 0$ | **0** | PASS |
| **Zero/Negative Remaining Time** | $\text{actual\_remaining\_minutes} \le 0$ | **0** | PASS |
| **Zero/Negative Scheduled Time** | $\text{scheduled\_time\_to\_go\_min} \le 0$ | **0** | PASS |
| **Negative Delays** | $\text{current\_delay\_min} < 0$ | **0** | PASS |
| **Excessive Runaway Delays** | $\text{current\_delay\_min} > 600\text{ min}$ (10h) | **0** (Max: $108.91\text{ min}$) | PASS |
| **Negative Speeds** | $\text{Implied Speed} < 0\text{ km/h}$ | **0** | PASS |
| **Zero Speeds** | $\text{Implied Speed} == 0\text{ km/h}$ | **0** | PASS |
| **Physically Impossible Speeds** | $\text{Implied Speed} > 200\text{ km/h}$ | **0** (Max: $110.01\text{ km/h}$) | PASS |

### Implied Speed Distribution
Calculated as:

$$v_{\text{implied}} = \frac{\text{distance\_to\_go\_km}}{\text{actual\_remaining\_minutes} / 60}$$

- **Minimum Implied Speed**: $6.99\text{ km/h}$ (occurs during heavy stationary signal halt or severe caution orders)
- **25th Percentile ($Q_1$)**: $89.23\text{ km/h}$
- **Median ($Q_2$)**: $100.20\text{ km/h}$
- **Mean**: $95.94\text{ km/h}$
- **75th Percentile ($Q_3$)**: $109.18\text{ km/h}$
- **Maximum Implied Speed**: $110.01\text{ km/h}$ (conforms with maximum passenger train speed limits on trunk lines)

---

## 7. Segment & Route Level Breakdown

### Breakdown by Train ID

| Train ID | Train Name | Total Samples | Mean Target (min) | Mean Delay (min) | Disrupted Samples | Disruption % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **12028** | Shatabdi Express | 625 | 27.08 | 14.98 | 215 | 34.40% |
| **12260** | Duronto Express | 875 | 84.11 | 19.53 | 280 | 32.00% |
| **12302** | Howrah Rajdhani Express | 875 | 69.12 | 21.13 | 315 | 36.00% |
| **12306** | Kolkata Rajdhani Express | 875 | 69.16 | 20.05 | 275 | 31.43% |
| **12610** | Chennai Express | 625 | 35.45 | 10.00 | 160 | 25.60% |
| **12926** | Paschim Express | 750 | 93.01 | 12.09 | 175 | 23.33% |
| **12952** | Mumbai Rajdhani Express | 750 | 76.85 | 15.14 | 230 | 30.67% |
| **12954** | August Kranti Tejas Rajdhani | 750 | 76.24 | 13.50 | 210 | 28.00% |

### Breakdown by Route ID

| Route ID | Route Corridor | Total Samples | Mean Target (min) | Mean Delay (min) | Disrupted Samples | Disruption % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | New Delhi – Howrah Eastern Trunk | 2,625 | 74.13 | 20.23 | 870 | 33.14% |
| **2** | New Delhi – Mumbai Central Western Trunk | 2,250 | 82.03 | 13.58 | 615 | 27.33% |
| **3** | KSR Bengaluru – MGR Chennai Central | 1,250 | 31.26 | 12.49 | 375 | 30.00% |

---

## 8. Conclusion & Recommendation

The synthetic dataset [`data/processed/synthetic_train_eta_dataset.csv`](file:///d:/Projects/dynamic-eta-forecasting/data/processed/synthetic_train_eta_dataset.csv) passes all quality criteria:
- **Completeness**: 0 nulls across 6,125 observations.
- **Diversity**: 8 passenger trains, 3 major corridors, 200 distinct journeys, 25 synthetic dates, 3 day types.
- **Physical Realism**: Speeds strictly bounded $[6.99, 110.01]\text{ km/h}$, realistic delay escalation up to 108 min, target correlated with physical distance and timetable.
- **Readiness**: The dataset is fully validated, completely isolated from unit test mutations, and ready for Day 2 Phase 2 machine learning model development.
