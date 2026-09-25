# SABARAGAMUWA UNIVERSITY OF SRI LANKA
## FACULTY OF COMPUTING - DEPARTMENT OF DATA SCIENCE
### CAPSTONE PROJECT IN DATA SCIENCE II (DS4105 / DS3206)

# DATA PIPELINE & TRAINING METHODOLOGY REPORT

| Field | Details |
|---|---|
| **Project Title** | Detecting AI-Generated Product Reviews: A Comparative Study of Linguistic, Statistical, and Transformer-Based Approaches |
| **Student Name / Index No.** | Sankajith D. Jinasena \| 22CDS0431 |
| **Internal Supervisor** | Dr. U.A.P. Ishanka |
| **Internal Co-Supervisor** | Miss H.K.P. Dineshika, Lecturer, Department of Computing and Information Systems |
| **Dataset** | Salminen et al. (2022) Amazon Fake Reviews Dataset - 40,405 labelled reviews, 10 categories |

---

## 1. Executive Summary & Overview

This technical report details the exact data preprocessing, feature representation, dataset partitioning, and model input specifications utilized across all four modeling approaches evaluated in this capstone project. 

The full cleaned dataset comprises **40,405 Amazon product reviews** balanced approximately 50% real (`OR`) and 50% fake (`CG`) across 10 distinct product categories.

### Dataset Partitioning & Feature Specification Matrix

| Modeling Approach | Input Data Representation | Feature Space Dimension | Training Set | Validation Set | Held-Out Test Set |
|---|---|---|---|---|---|
| **1. Engineered Features** | 16 Domain/Linguistic Numerical Metrics | 16 Dense | 32,324 (80%) | N/A | 8,081 (20%) |
| **2. TF-IDF Bag-of-Words** | Top 5,000 Unigrams & Bigrams | 5,000 Sparse | 32,324 (80%) | N/A | 8,081 (20%) |
| **3. Hybrid Fusion Model** | Stacked TF-IDF + Scaled Engineered Features | 5,016 Sparse/Dense | 32,324 (80%) | N/A | 8,081 (20%) |
| **4. Fine-Tuned DistilBERT** | Subword Tokens (`max_length=256`) | 768 Hidden Embeddings | 28,283 (70%) | 6,061 (15%) | 6,061 (15%) |

---

## 2. Dataset Pipeline & Cleaning

Prior to model training, raw data underwent standardized preprocessing:
1. **Deduplication & Null Removal**: Missing text strings and duplicate entries were filtered out, yielding 40,405 clean records.
2. **Label Standardisation**: Binary targets normalized to `fake` (computer-generated) and `real` (human-written).
3. **Text Cleaning Strategy**: Minimal text scrubbing (preserving casing, punctuation, and exclamation marks) was deliberately maintained because structural markers (ALL-CAPS ratio, punctuation density) carry discriminative signals for AI text detection.

---

## 3. Detailed Data Usage by Modeling Approach

### Approach 1: Hand-Engineered Features (Logistic Regression & Random Forest)
- **Data Input**: Rather than raw text tokens, 16 domain-specific numerical features were computed per review:
  - *Lexical Diversity*: `unique_word_ratio` (Type-Token Ratio)
  - *Readability Scores*: `flesch_reading_ease`, `flesch_kincaid_grade`
  - *Structural Metrics*: `word_count`, `char_count`, `avg_word_length`, `sentence_count`, `avg_sentence_length`
  - *Stylistic Density*: `punctuation_density`, `exclamation_ratio`, `capital_word_ratio`
  - *Sentiment Signals*: `sentiment_polarity`, `sentiment_subjectivity` (via TextBlob)
  - *Part-of-Speech (POS) Ratios*: `adjective_ratio`, `noun_ratio`, `verb_ratio` (via NLTK tagger)
- **Feature Scaling**: `StandardScaler` was fit strictly on the training set (mean=0, std=1) and transformed onto the test set to normalize feature magnitudes.
- **Partitioning**: Stratified **80% Train (32,324 samples)** / **20% Test (8,081 samples)** with `random_state=42`.

### Approach 2: TF-IDF Bag-of-Words (Logistic Regression & Linear SVM)
- **Data Input**: Raw `review_text` strings.
- **Vectorization Pipeline**:
  - `TfidfVectorizer` configured with `max_features=5000`, `ngram_range=(1, 2)` (unigrams and bigrams), `min_df=2`, and `stop_words='english'`.
  - Vocabulary fitting (`fit_transform`) executed strictly on training reviews to eliminate vocabulary leakage.
- **Partitioning**: Stratified **80% Train (32,324 samples)** / **20% Test (8,081 samples)** with `random_state=42`.

### Approach 3: Hybrid Fusion Model (Logistic Regression & Calibrated Linear SVM)
- **Data Input**: Dual-channel input fusing both raw text and numerical engineered features.
- **Feature Stacking Mechanics**:
  1. 5,000 TF-IDF sparse columns computed from `review_text`.
  2. 16 scaled dense engineered feature columns.
  3. Combined via horizontal sparse matrix concatenation (`scipy.sparse.hstack`), producing a **5,016-dimensional feature vector** per review.
- **Model Calibration**: Linear SVM wrapped with `CalibratedClassifierCV(cv=5)` to convert decision margins into calibrated prediction probabilities.
- **Partitioning**: Stratified **80% Train (32,324 samples)** / **20% Test (8,081 samples)**.

### Approach 4: Fine-Tuned DistilBERT Transformer
- **Data Input**: Raw `review_text` tokenized using HuggingFace `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
- **Tokenization & Tensor Formatting**:
  - `max_length=256` tokens (longer reviews truncated; shorter reviews zero-padded).
  - Encoded into PyTorch Tensors (`input_ids`, `attention_mask`, `labels`).
- **3-Way Data Partitioning**:
  - **70% Training Set (28,283 samples)**: Used to update 66M Transformer parameters via AdamW optimizer.
  - **15% Validation Set (6,061 samples)**: Evaluated at the end of each epoch for early stopping (`patience=2`) to prevent overfitting.
  - **15% Held-Out Test Set (6,061 samples)**: Kept completely isolated for final model evaluation.
- **Hyperparameters**: `batch_size=16`, `learning_rate=5e-5`, 3 training epochs, FP16 mixed precision on CUDA GPU.

---

## 4. Methodological Justification & Data Leakage Prevention

1. **Strict Train/Test Isolation**: All vectorizer vocabularies, scaler means/variances, and TF-IDF IDF values were learned *only* on training data before transforming validation/test sets.
2. **Stratified Sampling**: Class proportions (~50% real / ~50% fake) were preserved across all splits using stratified random sampling (`stratify=y`).
3. **Partition Alignment**: Classical models (80/20) and Transformers (70/15/15) both test on representative stratified held-out samples from the same underlying 40,405 population, ensuring methodological validity.
