# Predicting Persistence of Human Rabies Deaths Across CALABARZON Municipalities (2025 to 2026)

**Course:** CSST102 – Machine Learning Problem Formulation & Project Planning  
**Institution:** Laguna State Polytechnic University (LSPU) – BS Computer Science (BSCS 3-A)  
**Group:** Group 6  
**Members:**
* **John Kurt S. Larioza** (Group Leader)
* **Airelle Lambert E. Ramos**
* **Charles Lawrence R. Ferrer**

**Instructor:** Rolan Sotomayor  
**Data Source:** Department of Health (DOH) CHD IV-CALABARZON, *2026 Regional Updates on the Rabies Prevention and Control Program* (Slides 9–11 for 2025 and 2026 deaths as of September 5, 2026; context from Slides 3, 6, and 14).

---

## 1. Purpose of the Project

Human rabies remains a preventable yet fatal disease across **CALABARZON** (Cavite, Laguna, Batangas, Rizal, and Quezon). Although regional animal bite cases reached **503,058 in 2025** and **256,240 in Q1–Q2 2026**, municipalities that recorded human rabies deaths in 2025 or 2026 face recurring surveillance and prevention challenges alongside anti-rabies vaccine and ERIG supply constraints.

### Project Objectives
1. **Binary Persistence Classification (`target_death_2026`):** Predict whether a CALABARZON municipality/city in the dataset records at least one human rabies death in 2026 (`1`) or none (`0`) using 2025 death counts (`deaths_2025`), high-burden indicator (`high_2025`), LGU type (`is_city`), and `province`.
2. **Leakage-Free Preprocessing & Evaluation:** Build a reproducible `scikit-learn` `Pipeline` and `ColumnTransformer` with a stratified 70/30 train-test split (32 train / 14 test rows) and 5-fold stratified cross-validation while strictly excluding 2026 death counts from the input features to prevent target leakage.
3. **Transparent Model Comparison & Error Analysis:** Compare baseline, default, and tuned models across multiple metrics and document real-world limitations and bias before drawing operational conclusions.

---

## 2. Dataset Overview

* **Dataset File:** `calabarzon_municipal_rabies_deaths_2025_2026.csv`
* **Scope & Size:** **46 municipalities/cities** across CALABARZON that recorded at least one human rabies death in 2025 or 2026 (Cavite: 10, Laguna: 7, Batangas: 8, Rizal: 7, Quezon & Lucena City: 14).
* **Data Quality:** **0 missing values** and **0 duplicate rows**; total deaths match DOH regional reports (**39 in 2025** and **29 in 2026**). Note that Quezon's slide table prints a 2025 total of 8 while its individual rows sum to 7, and Lucena City (1 death in 2025) is listed separately; it is kept as its own row.
* **Target Balance (`target_death_2026`):** `1` (at least 1 death in 2026) = **25 LGUs (54.3%)**; `0` (0 deaths in 2026) = **21 LGUs (45.7%)**.

---

## 3. Performance Evaluation

Models were trained on **32 training rows** and evaluated on a **14-row stratified test split** (`random_state=42`), supplemented by **5-fold stratified cross-validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

### Test Set Performance Summary (14 Municipalities)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Most Frequent)** | 0.571 | 0.571 | 1.000 | 0.727 | 0.500 |
| **Logistic Regression (Baseline, `C=1.0`)** | 0.714 | 0.750 | 0.750 | 0.750 | 0.812 |
| **Random Forest (Baseline, `n_estimators=100`)** | 0.714 | 0.700 | 0.875 | 0.778 | 0.812 |
| **Logistic Regression (Tuned, `C=0.01`)** | 0.571 | 0.571 | 1.000 | 0.727 | 0.812 |
| **Random Forest (Tuned, `max_depth=3`, `min_samples_leaf=2`)** | **0.857** | **1.000** | **0.750** | **0.857** | **0.833** |

### Key Findings & Error Analysis
* **Tuned Random Forest Performance:** Constrained to `max_depth=3` and `min_samples_leaf=2` via `GridSearchCV` on the training set (Best CV ROC-AUC: **0.814**), the Tuned Random Forest achieved the highest test set **Accuracy (0.857)**, **Precision (1.000)**, **F1-Score (0.857)**, and **ROC-AUC (0.833)**, with a training accuracy of **0.875** (minimal overfitting).
* **Tuned Logistic Regression Collapse:** Strong regularization (`C=0.01`) caused Logistic Regression to predict mostly class `1`, dropping test accuracy to **0.571** (matching the majority-class dummy baseline).
* **Test Set Errors (2 / 14 Misclassified by Tuned Random Forest):**
  * **Confusion Matrix (`[[TN, FP], [FN, TP]]`):** `[[6, 0], [2, 6]]` — zero false positives, and **2 false negatives** (**Tanza, Cavite** and **Bacoor City, Cavite**, both had `deaths_2025 = 1` or `2` and `is_city = 1` / `high_2025 = 1` yet recorded 1 death in 2026).

---

## 4. Limitations, Bias & Ethical Safeguards

* **Small Sample & Selection Bias:** Only **46 rows** (and **14 test rows**, where 1 municipality equals ~7.1% accuracy) are included, covering only LGUs with at least one death in 2025 or 2026 rather than a random sample of all CALABARZON municipalities.
* **Missing Epidemiological Covariates:** The source slides lack municipal human/animal population denominators, bite case counts, and animal/human vaccination coverage rates, and 2026 figures only cover data up to September 5, 2026.
* **Operational Caution:** Because of small sample size, potential under-reporting, and missing population/vaccination variables, **this model must not be used on its own to guide vaccine or ERIG allocation**.

---

## 5. Repository Files

* `README.md` — Project purpose, dataset summary, performance evaluation, and limitations
* `CSST102_Rabies_Persistence_CALABARZON.ipynb` — Complete Google Colab notebook (Weeks 1–5)
* `calabarzon_municipal_rabies_deaths_2025_2026.csv` — 46-municipality dataset extracted from DOH CALABARZON slides 9–11
* `week5_test_predictions.csv` — Test set predictions and probabilities across models
* `week5_final_metrics.csv` — Summary evaluation metrics table
* `calabarzon_rabies_persistence_charts.png` — Visual summary of model metrics, confusion matrix, and province counts
* `Week-No1` to `Week-No5` `.docx` progress reports — Completed hardcopy progress reports for Weeks 1–5
