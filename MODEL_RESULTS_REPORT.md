# SABARAGAMUWA UNIVERSITY OF SRI LANKA
## FACULTY OF COMPUTING - DEPARTMENT OF DATA SCIENCE
### CAPSTONE PROJECT IN DATA SCIENCE II (DS4105 / DS3206)

# MODEL RESULTS & EVALUATION REPORT

| Field | Details |
|---|---|
| **Project Title** | Detecting AI-Generated Product Reviews: A Comparative Study of Linguistic, Statistical, and Transformer-Based Approaches |
| **Student Name / Index No.** | Sankajith D. Jinasena \| 22CDS0431 |
| **Internal Supervisor** | Dr. U.A.P. Ishanka |
| **Internal Co-Supervisor** | Miss H.K.P. Dineshika, Lecturer, Department of Computing and Information Systems |
| **Dataset** | Salminen et al. (2022) Amazon Fake Reviews Dataset - 40,405 labelled reviews, 10 categories |

---

## 1. Purpose

This document summarises the testing (held-out) performance of every model evaluated during this project, across four progressively complex modelling approaches: engineered linguistic features, TF-IDF, a fusion of both, and a fine-tuned transformer (DistilBERT). All figures reported below are measured on data not seen during training, evaluated across a **unified 70% Train / 15% Validation / 15% Test** partition scheme ($N = 6,061$ test reviews per model).

---

## 2. Full Comparative Results - All Models

**Table 1: Testing accuracy and F1-score for every model evaluated (Unified 70/15/15 Benchmark)**

| Approach | Model | Test Set Size | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) |
|---|---|---|---|---|---|---|
| **Engineered Features** | Logistic Regression | 6,061 (15% split) | 77.92% | 77.96% | 77.81% | 0.779 |
| **Engineered Features** | Random Forest | 6,061 (15% split) | 85.27% | 85.36% | 85.11% | 0.852 |
| **TF-IDF** | Logistic Regression | 6,061 (15% split) | 87.82% | 88.71% | 86.66% | 0.877 |
| **TF-IDF** | Linear SVM | 6,061 (15% split) | 87.82% | 88.25% | 87.26% | 0.877 |
| **Fusion (TF-IDF + Engineered)** | Logistic Regression | 6,061 (15% split) | 90.15% | 90.59% | 89.60% | 0.901 |
| **Fusion (TF-IDF + Engineered)** | Linear SVM (Calibrated) | 6,061 (15% split) | 90.89% | 91.88% | 89.70% | 0.908 |
| **Fine-tuned Transformer** | DistilBERT | 6,061 (15% split) | **98.05%** | **97.03%** | **99.14%** | **0.981** |

---

### Accuracy Progression Across the Four Modelling Approaches (Best Result per Approach)

```
DistilBERT (fine-tuned)             |========================================| 98.05%
Fusion Model (TF-IDF + Engineered)  |===================================| 90.89%
TF-IDF (Linear SVM)                 |=================================| 87.82%
Engineered Features (Random Forest) |==============================| 85.27%
                                    0%        20%       40%       60%       80%       100%
```

---

## 3. Best Model: Fine-Tuned DistilBERT

The fine-tuned DistilBERT transformer achieved the strongest overall performance, evaluated on 6,061 held-out test reviews:

| Metric | Value |
|---|---|
| **Accuracy** | **98.05%** |
| **F1-score (fake class)** | **0.981** (0.9807) |
| **Precision (fake class)** | **97.03%** |
| **Recall (fake class)** | **99.14%** |

---

### Confusion Matrix for the Fine-Tuned DistilBERT Model

| | **Predicted Real** | **Predicted Fake** |
|---|---|---|
| **Actual Real** | **2,941** | **92** (False Positive) |
| **Actual Fake** | **26** (False Negative) | **3,002** |

> **Summary**: Out of 6,061 test reviews, only **118 were misclassified** (92 false positives, 26 false negatives), giving a test error rate of approximately **1.95%**.

---

## 4. Feature Importance (Engineered Features Model)

For the engineered-features Random Forest model, the most predictive individual features were consistent with the exploratory data analysis findings — length and readability signals dominated over sentiment and punctuation-based features.

**Top Feature Importances (Random Forest):**

| Feature Name | Description / Signal | Importance Weight |
|---|---|---|
| `unique_word_ratio` | Type-Token Ratio (Lexical Diversity) | **0.1665** |
| `flesch_kincaid_grade` | Readability Grade Level | **0.1091** |
| `avg_word_length` | Average Character Count per Word | **0.1089** |
| `char_count` | Total Character Length | **0.0839** |
| `avg_sentence_length` | Average Words per Sentence | **0.0771** |
| `flesch_reading_ease` | Readability Score | **0.0682** |
| `word_count` | Total Word Count | **0.0635** |
| `punctuation_density` | Punctuation Character Density | **0.0537** |
| `adjective_ratio` | POS Ratio (NLTK) | **0.0467** |
| `sentiment_polarity` | TextBlob Polarity (-1 to +1) | **0.0454** |
| `noun_ratio` | POS Ratio (NLTK) | **0.0430** |
| `sentiment_subjectivity` | TextBlob Subjectivity (0 to 1) | **0.0415** |
| `verb_ratio` | POS Ratio (NLTK) | **0.0385** |

---

## 5. Summary of Findings

- **Accuracy improved consistently with model complexity**: **85.27%** (best engineered-features model) $\rightarrow$ **87.82%** (best TF-IDF model) $\rightarrow$ **90.89%** (best fusion model) $\rightarrow$ **98.05%** (fine-tuned DistilBERT).
- **The fusion model (90.89% accuracy / 0.908 F1)** outperformed TF-IDF alone (**87.82% accuracy / 0.877 F1**) by **3.07 percentage points in accuracy and ~0.031 in F1-score**, indicating that engineered linguistic features carry complementary signal beyond raw word frequency.
- **The fine-tuned transformer achieved an ~7.16 percentage point accuracy gain** over the best classical fusion model (and **~10.23 points** over TF-IDF), attributable to its sensitivity to subtle generation artifacts that simpler feature representations do not capture.
- **Recall on the fake class (99.14% for DistilBERT)** was prioritised as the most operationally important metric, since failing to flag a fake review (false negative) is the more costly error in a trust-and-safety context.
