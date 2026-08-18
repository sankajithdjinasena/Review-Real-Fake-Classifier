# Detecting AI-Generated Product Reviews

A comparative study of linguistic, statistical, and transformer-based approaches to detecting AI-generated (fake) product reviews, built for the Capstone Project in Data Science II (DS4105), Department of Data Science, Faculty of Computing, Sabaragamuwa University of Sri Lanka.

This project compares four modelling approaches — hand-engineered linguistic features, TF-IDF, a fusion of the two, and a fine-tuned DistilBERT transformer — for classifying Amazon product reviews as genuine or AI-generated, and ships a working interactive demo alongside the analysis.

## Results at a glance

| Approach | Model | Accuracy | F1 (fake) |
|---|---|---|---|
| Engineered Features | Random Forest | 82.75% | 0.824 |
| TF-IDF | Linear SVM | 88.39% | 0.884 |
| Fusion (TF-IDF + Engineered) | Linear SVM | 90.36% | 0.903 |
| **Fine-tuned DistilBERT** | Transformer | **98.30%** | **0.983** |


## Dataset

[Salminen et al. (2022)](https://doi.org/10.1016/j.jretconser.2021.102771) Amazon Fake Reviews Dataset — 40,405 reviews across 10 product categories, near-balanced between genuine reviews and GPT-2-generated fakes.

Source: https://osf.io/tyue9/ (also mirrored on Kaggle as "Fake Reviews Dataset")

Expected raw columns: `category`, `rating`, `label`, `text_`

## Project structure

```
.
├── data/
│   ├── raw/
│   │   └── fake_reviews_dataset.csv
│   ├── processed/
│   │   ├── fake_reviews_cleaned.csv
│   │   ├── fake_reviews_features.csv
│   │   ├── fake_reviews_features_full.csv
│   │   └── fake_reviews_merged.csv
│   ├── eda_outputs/
│   │   ├── category_distribution.png
│   │   ├── class_balance.png
│   │   ├── rating_by_label.png
│   │   └── review_length_by_label.png
│   └── misclassified_reviews.csv
├── src/
│   ├── bert_fake_reviews_model/        # saved fine-tuned model
│   ├── app.py                          # dashboard / live classifier
│   ├── fake_reviews_bert_finetune.py
│   └── fake_reviews_error_analysis.py
├── 01_fake_reviews_clean_eda.ipynb
├── 02_fake_reviews_feature_engineering.ipynb
├── 03_fake_reviews_baseline_models.ipynb
├── 04_fake_reviews_fusion_model.ipynb
├── BERT_Model_and_Error_Analysis_Run_script.ipynb
├── .gitignore
└── README.md

```
## Setup

```bash
pip install pandas numpy scikit-learn scipy nltk textstat textblob \
            torch transformers streamlit plotly flask flask-cors \
            matplotlib seaborn openpyxl --break-system-packages

python -m textblob.download_corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"
```

A CUDA-capable GPU is strongly recommended for Step 5 (transformer fine-tuning); the rest of the pipeline runs fine on CPU.

## Pipeline: how to reproduce the results

Run in order. Each step's output feeds the next.

**1. Clean the data and explore it**

Open and run `01_fake_reviews_clean_eda.ipynb`, pointing it at `data/raw/fake_reviews_dataset.csv`.
→ produces `data/processed/fake_reviews_cleaned.csv` and plots in `data/eda_outputs/`.

**2. Engineer linguistic features**

Open and run `02_fake_reviews_feature_engineering.ipynb`, reading `data/processed/fake_reviews_cleaned.csv` and writing `data/processed/fake_reviews_features_full.csv`.
Run on a small sample first (e.g. 2,000 rows) for a quick test before committing to the full dataset — the POS-tagging step is slow.

**3. Train and compare the classical baselines**

Open and run `03_fake_reviews_baseline_models.ipynb`, which trains and compares:
- the engineered-features model, using `data/processed/fake_reviews_features_full.csv`
- the TF-IDF model, using `data/processed/fake_reviews_cleaned.csv`

**4. Train the fusion model**

First merge the text and engineered features into one file (must come from the same non-sampled run):
```python
import pandas as pd
cleaned = pd.read_csv("data/processed/fake_reviews_cleaned.csv")
feats = pd.read_csv("data/processed/fake_reviews_features_full.csv")
merged = pd.concat([cleaned[["review_text"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
merged.to_csv("data/processed/fake_reviews_merged.csv", index=False)
```
Then open and run `04_fake_reviews_fusion_model.ipynb`, using `data/processed/fake_reviews_merged.csv`.

**5. Fine-tune DistilBERT** (GPU recommended)
```bash
python src/fake_reviews_bert_finetune.py \
    --input data/processed/fake_reviews_cleaned.csv \
    --model_name distilbert-base-uncased \
    --epochs 3 \
    --batch_size 16 \
    --output_dir ./src/bert_fake_reviews_model
```
Test with `--sample 5000` first to validate the pipeline before the full run.

Alternatively, run the combined `BERT_Model_and_Error_Analysis_Run_script.ipynb`, which fine-tunes DistilBERT and runs the error analysis below in one pass.

**6. Analyze the model's errors**
```bash
python src/fake_reviews_error_analysis.py \
    --input data/processed/fake_reviews_cleaned.csv \
    --model_dir ./src/bert_fake_reviews_model \
    --output_csv data/misclassified_reviews.csv
```

## Interactive dashboard + live classifier

The easiest way to explore the results and try the live model:

```bash
streamlit run src/app.py
```

Set `DEFAULT_MODEL_DIR` at the top of the script (or use the sidebar path input) to point at your `src/bert_fake_reviews_model` folder. Opens at `http://localhost:8501` with sections for the dataset EDA, model comparison, confusion matrix, error analysis case notes, and a live "paste a review, get a prediction" interface.

## Key findings

- AI-generated reviews in this dataset are systematically **shorter, less lexically diverse, and structurally simpler** than genuine reviews (73.6 vs. 61.3 words on average); star rating carries no discriminative signal.
- Combining TF-IDF with engineered linguistic features (**fusion model**) meaningfully outperforms either alone, confirming the two feature families capture complementary signal.
- The fine-tuned transformer's ~8-point accuracy gain over the best classical model is attributable to its sensitivity to GPT-2-specific generation artifacts — not a generic "deep learning is better" effect.
- Qualitative error analysis found the model's mistakes are systematic: it tends to flag unusually brief genuine reviews as fake, and is fooled by fake reviews that mimic genuine review conventions (structured pros/cons lists, disclosure language).
- **Important limitation:** reported accuracy likely reflects the comparative ease of detecting a 2019-era generator (GPT-2); performance against modern LLM-generated reviews is untested.

## References

Full IEEE-formatted reference list is in the final report. Key sources:

- Salminen, J., Kandpal, C., Kamel, A. M., Jung, S., & Jansen, B. J. (2022). Creating and detecting fake reviews of online products. *Journal of Retailing and Consumer Services*, 64, 102771.
- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. *ACL-HLT 2011*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.

## License / Academic Use

Built as an individual capstone project (DS4105) at Sabaragamuwa University of Sri Lanka, 2025/2026. Dataset used under its original license terms (Salminen et al., 2022).