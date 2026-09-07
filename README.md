# Data Science & Machine Learning Exploratory Data Analysis (EDA)

A centralized repository containing comprehensive Exploratory Data Analysis (EDA), statistical validation, data preprocessing, and initial modeling across five distinct real-world and synthetic datasets.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Datasets Summary](#datasets-summary)
3. [Environment & Dependencies](#environment--dependencies)
4. [Dataset Analyses & Findings](#dataset-analyses--findings)
   - [1. King County House Sales Analysis](#1-king-county-house-sales-analysis)
   - [2. Multi-Class Weather Classification](#2-multi-class-weather-classification)
   - [3. Social Network Ads Purchase Classification](#3-social-network-ads-purchase-classification)
   - [4. Teen Social Media & Mental Health Analysis](#4-teen-social-media--mental-health-analysis)
   - [5. Cardiovascular Disease Risk Prediction (2026)](#5-cardiovascular-disease-risk-prediction-2026)
5. [Key Takeaways & Modeling Guidelines](#key-takeaways--modeling-guidelines)
6. [How to Run](#how-to-run)

---

## Project Overview

This collection covers end-to-end data workflows for regression, binary classification, and multiclass classification tasks:
* **Data Auditing:** Identifying missing values, structural duplicates, and data types.
* **Feature Engineering:** Deriving domain-specific metrics (e.g., temporal breakdowns, ratios, interaction terms).
* **Exploratory Data Analysis:** Distribution assessment, outlier detection via Interquartile Range (IQR), correlation checks, and target-stratified comparisons.
* **Statistical Hypothesis Testing:** Independent two-sample t-tests, Pearson correlation coefficients, Chi-square tests of independence ($\chi^2$), and Multivariate Analysis of Variance (MANOVA).
* **Predictive Prototyping:** Baseline modeling, hyperparameter/feature selection, scaling strategies (`RobustScaler`), and cross-validation.

---

## Datasets Summary

| # | Dataset / Domain | Target Variable | Problem Type | Samples | Features | Key Highlights |
|---|---|---|---|---|---|---|
| **1** | [King County House Sales](#1-king-county-house-sales-analysis) | `price` | Regression | 4,600 | 18 (24 post-FE) | Temporal parsing, real estate ratios, right-skewed pricing |
| **2** | [Weather Observations](#2-multi-class-weather-classification) | `normalized_label` (0–3) | Multiclass Classification | 533,312 | 8 (9 post-FE) | Massive volume, severe class imbalance, distinct wind/radiation clusters |
| **3** | [Social Network Ads](#3-social-network-ads-purchase-classification) | `Purchased` (0/1) | Binary Classification | 400 | 5 | Logistic Regression, `RobustScaler`, coefficient feature importance |
| **4** | [Teen Mental Health (Synthetic)](#4-teen-social-media--mental-health-analysis) | `depression_label` (0/1) | Binary Classification (Imbalanced) | 1,200 | 13 | T-tests ($p < 10^{-9}$), MANOVA (Wilks' $\Lambda$), Chi-square evaluation |
| **5** | [Cardiovascular Disease Risk](#5-cardiovascular-disease-risk-prediction-2026) | `has_heart_disease` (0/1) | Binary Classification | 9,000 | 27 | Clinical & lifestyle markers, clean baseline, balanced features |


## Dataset Analyses & Findings

### 1. King County House Sales Analysis

* **File/Source:** `data.csv` (`shree1992/housedata`)
* **Objective:** Predict real estate sale prices and assess structural/environmental drivers.
* **Data Preparation & Engineering:**
  * Cleaned 4,600 rows $\times$ 18 columns with 0 null values and 0 duplicate records.
  * Extracted temporal components: parsed `date` to datetime, deriving `Year` and `Month`.
  * Computed domain metrics:
    * `House_Age = Current_Year - yr_built`
    * `renovated = (yr_renovated != 0).astype(int)`
    * `Price_per_sqft = price / sqft_living`
    * `lot_utilization = sqft_living / sqft_lot`
  * Sanity checks verified no negative values across geometric, monetary, or count metrics.
* **Key EDA Observations:**
  * **Price Distribution:** Highly right-skewed with extreme values representing luxury properties; lower-to-mid range homes dominate the distribution mass.
  * **Square Footage vs. Price:** Strong, positive linear relationship between `sqft_living` and `price`.
  * **Waterfront & View:** Homes with a waterfront location (`waterfront == 1`) or higher view ratings command significantly higher median prices and wider interquartile ranges.
  * **Bedrooms:** General upward price trend as bedroom counts increase up to 5–6 rooms, after which variance increases sharply.

---

### 2. Multi-Class Weather Classification

* **File/Source:** `Combined12.csv` (`jaydeepvaishya/weather-dataset`)
* **Objective:** Categorize weather conditions into 4 discrete classes (`normalized_label`: `0, 1, 2, 3`).
* **Data Quality Audit:**
  * 533,312 records across 8 numerical features.
  * Zero missing fields across the entire dataset.
  * 16,264 identical/duplicate rows identified. These duplicates were retained because weather stations frequently record identical readings across different observation timestamps.
  * Derived `temp_range = temp_max(c) - temp_min(c)` to capture diurnal temperature variance.
* **Class Profiles & Separation:**
  * Class counts exhibit noticeable imbalance:
    * `Class 0`: 239,241
    * `Class 1`: 155,709
    * `Class 2`: 87,533
    * `Class 3`: 50,829
  * **Atmospheric Pressure:** Ineffective for separation; distributions and boxplot whiskers almost entirely overlap (interquartile ranges cluster around 1008–1020 hPa).
  * **Global Radiation & Temperature:** Effectively separates into two temperature regimes:
    * `Classes 0 & 2`: Lower temperatures, lower global solar radiation.
    * `Classes 1 & 3`: Elevated temperatures, significantly higher solar radiation.
  * **Wind Speed:** Separates observations orthogonally to temperature:
    * `Classes 0 & 1`: Moderate/calm wind speeds.
    * `Classes 2 & 3`: High wind speeds.

---

### 3. Social Network Ads Purchase Classification

* **File/Source:** `Social_Network_Ads.csv` (`dragonheir/logistic-regression`)
* **Objective:** Predict consumer ad conversion (`Purchased`: `0` = No, `1` = Yes).
* **Data Pipeline & Training:**
  * 400 entries with no missing or duplicate records.
  * One-hot encoded `Gender` (`Gender_Female`, `Gender_Male`).
  * Features evaluated: `Age`, `EstimatedSalary`, `Gender`, `User ID`.
  * Scaled feature matrices using `RobustScaler` to limit outlier leverage.
  * Applied `LogisticRegression(max_iter=1000)` evaluated on an 80/20 train/test split and 5-fold cross-validation.
* **Performance & Feature Importance:**
  * **Test Metrics:** 88.75% Test Accuracy, Macro F1-score of 0.87 (0.92 for Class 0, 0.82 for Class 1).
  * **Cross-Validation:** 5-fold CV Mean Accuracy of **81.56%**.
  * **Model Coefficients:**

| Feature | Coefficient | Absolute Value | Impact |
|---|---|---|---|
| `Age` | +2.7307 | 2.7307 | Dominant driver of conversion |
| `EstimatedSalary` | +1.4163 | 1.4163 | Secondary positive driver |
| `Gender_Female` / `Male` | $\mp 0.1238$ | 0.1238 | Marginal effect |
| `User ID` | +0.0290 | 0.0290 | Insignificant |

---

### 4. Teen Social Media & Mental Health Analysis

* **File/Source:** `Teen_Mental_Health_Dataset.csv` (`sunil123kumar/social-media-impact-on-mental-health`)
* **Objective:** Assess how social media metrics and lifestyle factors correlate with adolescent depression (`depression_label`).
* **Data Composition:**
  * 1,200 participants, 13 features; 0 missing values, 0 duplicate records.
  * Heavily skewed target: 1,169 non-depressed (`0`) vs. 31 depressed (`1`).
* **Statistical Hypothesis Testing:**
  * **Gender Disparity:** $\chi^2$ test of independence yielded $\chi^2 = 0.255$, $p = 0.613$. Gender shows no statistically significant association with depression status in this cohort.
  * **Sleep Deprivation:** Two-sample t-test yielded $t = 6.721$, $p = 2.78 \times 10^{-11}$. Depressed teens average significantly fewer hours of sleep.
  * **Daily Usage:** Two-sample t-test yielded $t = -6.159$, $p = 9.96 \times 10^{-10}$. Depressed individuals average significantly higher daily social media screen time.
  * **Screen Time Before Bed:** $t = 0.571$, $p = 0.568$ (not statistically significant on its own).
  * **Academic Performance & Physical Activity:** $p = 0.960$ and $p = 0.543$ respectively; neither is statistically significant in isolation.
  * **Multivariate Evaluation (MANOVA):**
    * Model: `[features] ~ depression_label`
    * Wilks' Lambda: $\Lambda = 0.8798$, $F = 20.32$, $p < 0.0001$. Confirms joint multivariate significance across psychological and behavioral features.

---

### 5. Cardiovascular Disease Risk Prediction (2026)

* **File/Source:** `heart_disease_risk_2026.csv` (`uditjain13/heart-disease-risk-2026`)
* **Objective:** Identify high-risk cardiovascular disease profiles (`has_heart_disease`).
* **Data Hygiene & Attributes:**
  * 9,000 patient records $\times$ 27 attributes.
  * Zero missing values; zero duplicate rows.
  * Dropped uninformative identifier: `patient_id`.
  * **Target Distribution:** 6,273 Negative (`0`, 69.7%) vs. 2,727 Positive (`1`, 30.3%).
* **Feature Split:**
  * **Numerical (20):** Clinical lab markers (`cholesterol_total`, `hdl`, `ldl`, `triglycerides`, `fasting_blood_sugar`, `hba1c`), vitals (`resting_bp_systolic`, `resting_bp_diastolic`, `resting_heart_rate`, `max_heart_rate_achieved`), and lifestyle metrics (`exercise_minutes_per_week`, `sleep_hours`, `stress_score`, `daily_steps`, `diet_quality_score`, etc.).
  * **Categorical (6):** `sex`, `chest_pain_type`, `exercise_induced_angina`, `family_history`, `smoker_status`, `wearable_owner`.

## Key Takeaways & Modeling Guidelines

1. **Address Target Imbalance First:**
   Datasets like Teen Mental Health (`depression_label` at ~2.6% positive) and Weather Classification require stratified splits, class-weighted loss functions (`class_weight='balanced'`), or resampling strategies (SMOTE / Random Under-Sampling) alongside precision-recall curves rather than raw accuracy.
2. **Handle Extreme Skewness:**
   Target variables such as real estate `price` require log transforms ($\log(1+y)$) when fitting linear models to stabilize variance and normalize residuals.
3. **Filter Redundant Features:**
   In weather classification, features like atmospheric `Pressure` show minimal between-class variance and should be candidates for removal to reduce dimensionality.
4. **Choose the Right Scaler:**
   When working with skewed data containing valid outliers (such as income, housing dimensions, or step counts), `RobustScaler` (scaling via median and IQR) performs better than standard z-score normalization.
