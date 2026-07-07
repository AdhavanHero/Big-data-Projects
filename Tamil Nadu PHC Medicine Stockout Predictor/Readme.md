# 🏥 Tamil Nadu PHC Medicine Stockout Predictor

**Predicting medicine stockout risk across Tamil Nadu's Primary Health Centres using collaborative filtering (ALS) on government health data.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Primary Health Centres (PHCs) across Tamil Nadu regularly run out of essential medicines and vaccines — ORS, IFA tablets, Zinc, Albendazole, and routine immunization vaccines (BCG, DPT, OPV, TT, Hepatitis B) — often without any early warning. By the time a stockout is noticed, patients are already being turned away.

This project asks a simple question:

> **Can we predict, one month in advance, which medicine is most likely to run out in which district — using only the stock records the government already collects?**

The intuition borrows directly from recommendation systems: instead of predicting *"which movie will this user like,"* the model predicts *"which medicine is this district about to run out of."*

```
Netflix:      User     x Movie     x Rating       → Recommend movies
This project: District x Medicine  x Utilization   → Predict stockout risk
```

---

## 📊 Dataset

**Source:** [HMIS (Health Management Information System) – data.gov.in](https://data.gov.in), Government of India

The HMIS "Item Report" publishes month-wise stock and distribution figures for essential medicines and vaccines at the district level, split by facility type (Public/Private) and area (Urban/Rural).

| Fiscal Year | File | Approx. Size |
|---|---|---|
| 2013–14 | `hmis-item-rpt-tn-for-2013-14.csv` | ~2.2 MB |
| 2014–15 | `hmis-item-rpt-tn-for-2014-15.csv` | ~2.2 MB |
| 2016–17 | `hmis-item-rpt-tn-for-2016-17.csv` | ~2.2 MB |
| 2018–19 | `hmis-item-rpt-tn-for-2018-19.csv` | ~19.9 MB |
| 2019–20 | `hmis-item-rpt-tn-for-2019-20.csv` | ~4.8 MB |

Each raw file is in **wide format**: one row per `(district, parameter, record_type)` with 12 monthly columns (April → March) plus a yearly total, further broken down into Total / Public / Private / Urban / Rural sub-columns.

**Combined after loading:** 32 districts × 5 fiscal years → **77,888 rows**, 18 columns after standardization.

> ⚠️ Files are read with `encoding="iso-8859-1"` — the raw government CSVs are not UTF-8 and will throw decode errors otherwise.

---

## 🧠 Methodology

### 1. Load & Clean
All 5 yearly files are read independently (schemas differ slightly by year — a `SIMPLE_YEARS` vs `RICH_YEARS` loader handles that), tagged with their fiscal year, and unioned into one master DataFrame. Duplicates are dropped, string columns are trimmed, and known placeholder values (`"NA"`, `""`, `"Data for this item can not be cumulated"`) are nulled out.

### 2. Feature Engineering — Wide → Long
The dataset is filtered down to **12 key medicines/vaccines** (ORS, IFA tablets, Albendazole, Zinc, Calcium, and the OPV/BCG/DPT/TT/Hepatitis-B vaccines), then reshaped using Spark's `stack()` function:

- **Wide format:** 1 row = 1 year of stock (12 month columns)
- **Long format:** 1 row = 1 district–medicine–month observation

This is done twice — once for `"5. Total Stock"` and once for `"4. Stock Distributed"` — and the two are joined together.

### 3. Utilization Score & Proxy Stockout Label
There's no explicit "stockout" flag in the raw data, so a proxy label is engineered:

```
Utilization Score = Stock Distributed / (Total Stock + 1)

Score → 1.0  = almost all stock consumed  → HIGH risk
Stock == 0   = confirmed stockout         → score forced to 1.0

Label: stockout_risk = 1 (HIGH) if stock == 0 OR utilization ≥ 0.85
                      = 0 (LOW)  otherwise
```

Across the training data, this produces a fairly balanced label distribution: **~42% HIGH risk, ~58% LOW risk**.

### 4. Encoding for ALS
ALS requires integer IDs, not strings, so:
- `user_key = district + fiscal_year` (so the same district in different years is treated as a distinct "user" — capturing how risk evolves over time)
- `StringIndexer` converts `user_key` and `parameter` (medicine) into `user_index` / `item_index`

### 5. Model — ALS (Alternating Least Squares)
Implemented as a `Pipeline` (`StringIndexer → ALS`) using PySpark MLlib:

| Parameter | Value | Why |
|---|---|---|
| `rank` | 50 | number of latent factors per district/medicine |
| `maxIter` | 15 | training iterations |
| `regParam` | 0.1 | regularization, prevents overfitting |
| `implicitPrefs` | `True` | utilization is an implicit signal, not an explicit 1–5 rating |
| `coldStartStrategy` | `drop` | ignore unseen district/medicine combos at test time |

**Split:** 80/20 train-test (6,818 / 1,723 rows), `seed=42`.

### 6. Evaluation & Output
The model is evaluated with **RMSE** and **MAE** against the held-out test set, then used to generate the final deliverable via `recommendForAllUsers(5)`: **the top-5 highest-risk medicines per district**, ranked into a `HIGH` / `LOW` risk report a health officer could act on monthly.

---

## 📈 Results

```
Rows loaded         : 77,888  (32 districts × 5 fiscal years)
Train / Test split  : 6,818 / 1,723  (80/20)
User factors        : 64   (district × fiscal-year combinations)
Item factors        : 12   (key medicines/vaccines)
Training time        : ~7 seconds

RMSE  : 1.8954
MAE   : 1.7153

Total HIGH-risk district–medicine pairs flagged : 320
```

**Sanity check passed:** average predicted score for `stockout_risk = 1` rows is consistently higher than for `stockout_risk = 0` rows — confirming the model is learning a genuine risk signal rather than noise.

**Sample output (Ariyalur district, top predicted risks):**

| District | Fiscal Year | Medicine | Risk Score | Level |
|---|---|---|---|---|
| Ariyalur | 2019-20 | Albendazole 400 mg tablet | 2.69 | HIGH |
| Ariyalur | 2019-20 | Zinc 20 mg tablet | 2.63 | HIGH |
| Ariyalur | 2019-20 | Vaccine - Hepatitis B | 2.52 | HIGH |
| Ariyalur | 2019-20 | ORS (New WHO) | 2.42 | HIGH |

### ⚠️ Honest limitation
The RMSE (1.90) is well above the initial target of <0.5. This is a known limitation of the current prototype, most likely because:
- Implicit-feedback ALS is being scored with a regression metric (RMSE) it wasn't directly optimized for
- Only 12 item factors and 64 user factors means very little data per latent dimension at `rank=50`
- The proxy label (0.85 utilization threshold) is a heuristic, not ground-truth stockout data

The **ranking output** (top-5 riskiest medicines per district) is still directionally useful and passes the sanity check above, but the raw RMSE should be read as a prototype-stage number, not a production metric. See [Future Work](#-future-work) below.

---

## 🛠️ Tech Stack

- **PySpark 4.1.2** (MLlib: `Pipeline`, `StringIndexer`, `ALS`, `RegressionEvaluator`)
- **Java 17** (required runtime for this PySpark version)
- Local execution on Windows (i7-12700H, 16GB RAM, RTX 3060) — moved off Databricks Community Edition, which couldn't handle this workload reliably
- Model persistence via `pickle` + CSV export of ALS user/item factors (a portable alternative to Spark's native `PipelineModel.save()`, which ran into Hadoop/`winutils` issues on Windows)

---

## 📁 Project Structure

```
.
├── data/
│   ├── hmis-item-rpt-tn-for-2013-14.csv
│   ├── hmis-item-rpt-tn-for-2014-15.csv
│   ├── hmis-item-rpt-tn-for-2016-17.csv
│   ├── hmis-item-rpt-tn-for-2018-19.csv
│   └── hmis-item-rpt-tn-for-2019-20.csv
├── ML_Pipeline_TN_PHC_Stockout.ipynb   # main pipeline: load → clean → feature engineer → train → evaluate
├── model/                              # saved ALS user/item factors (pickle + CSV)
└── README.md
```

---

## ▶️ How to Run

1. Install Java 17 and set `JAVA_HOME`
2. Install PySpark:
   ```bash
   pip install pyspark==4.1.2
   ```
3. Place all 5 HMIS CSVs in a `data/` folder and update the file paths in the notebook
4. Run the notebook top to bottom — Sections 1 through 10 execute the full pipeline sequentially

---

## 🚧 Future Work

- Replace the heuristic 0.85-utilization proxy label with a supervised classifier (e.g. Gradient Boosted Trees) trained directly to predict stockout, using ALS-derived latent factors as input features
- Add district-level covariates (population, PHC count, rainfall/seasonality) to reduce reliance on stock data alone
- Backtest predictions against known real-world stockout events, if such records can be sourced
- Extend from 5 non-contiguous fiscal years to a continuous monthly time series for proper time-aware validation (currently a random 80/20 split, not a temporal split)

---

## 👤 Author

**Adhavan**
Built as part of a personal data science portfolio, applying PySpark MLlib collaborative filtering to public health supply-chain data for Tamil Nadu's ~2,780 PHCs.

**Dataset credit:** HMIS, Ministry of Health & Family Welfare, Government of India (via [data.gov.in](https://data.gov.in))
