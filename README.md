# Detecting AI-Generated Product Reviews

A comparative study of linguistic, statistical, and transformer-based approaches to detecting AI-generated (fake) product reviews, built for the Capstone Project in Data Science II (DS4105), Department of Data Science, Faculty of Computing, Sabaragamuwa University of Sri Lanka.

This project compares four modelling approaches - hand-engineered linguistic features, TF-IDF, a fusion of the two, and a fine-tuned DistilBERT transformer - for classifying Amazon product reviews as genuine or AI-generated, and ships an interactive **React & FastAPI dashboard** alongside the analysis.

## Results at a glance

| Approach | Model | Accuracy | F1 (fake) |
|---|---|---|---|
| Engineered Features | Random Forest | 85.27% | 0.852 |
| TF-IDF | Linear SVM | 87.82% | 0.877 |
| Fusion (TF-IDF + Engineered) | Linear SVM | 90.89% | 0.908 |
| **Fine-tuned DistilBERT** | Transformer | **98.05%** | **0.981** |

---

## Quick Start (Run the React Dashboard)

Launch the backend REST API server and React frontend with a single command:

```bash
python run.py
```
Open **http://127.0.0.1:8000** in your browser.

---

## Project Structure

```
.
├── backend/
│   ├── api.py                      # FastAPI REST API server
│   ├── app_streamlit.py            # Legacy Streamlit app backup
│   ├── modules/                    # Feature engineering & inference code
│   │   ├── feature_engineering.py
│   │   ├── fusion_inference.py
│   │   ├── train_fusion_model.py
│   │   ├── fake_reviews_bert_finetune.py
│   │   └── fake_reviews_error_analysis.py
│   └── models/                     # Saved fine-tuned BERT and Fusion models
│       ├── bert_fake_reviews_model/
│       └── fusion_model/
├── frontend/
│   ├── index.html                  # React SPA HTML entry point
│   ├── vite.config.js              # Vite configuration
│   ├── package.json                # React, Recharts, Tailwind dependencies
│   └── src/                        # React components & UI code
├── notebooks/                      # Jupyter notebooks & reproduction scripts
│   ├── 01_fake_reviews_clean_eda.ipynb
│   ├── 02_fake_reviews_feature_engineering.ipynb
│   ├── 03_fake_reviews_baseline_models.ipynb
│   ├── 04_fake_reviews_fusion_model.ipynb
│   └── BERT_Model_and_Error_Analysis_Run_script.ipynb
├── data/                           # Data files & misclassification CSVs
├── requirements.txt                # Python environment requirements
├── run.py                          # 1-Click launcher script
└── README.md
```

---

## Setup & Dependencies

```bash
pip install -r requirements.txt
```

To run the React frontend in dev mode with hot reloading:
```bash
cd frontend
npm install
npm run dev
```

---

## Pipeline: how to reproduce the results

Run in order. Each step's output feeds the next.

**1. Clean the data and explore it**
Run `notebooks/01_fake_reviews_clean_eda.ipynb`.

**2. Engineer linguistic features**
Run `notebooks/02_fake_reviews_feature_engineering.ipynb`.

**3. Train classical baselines**
Run `notebooks/03_fake_reviews_baseline_models.ipynb`.

**4. Train fusion model**
Run `backend/modules/train_fusion_model.py` or run `notebooks/04_fake_reviews_fusion_model.ipynb`.

**5. Fine-tune DistilBERT** (GPU recommended)
```bash
python backend/modules/fake_reviews_bert_finetune.py \
    --input data/processed/fake_reviews_cleaned.csv \
    --model_name distilbert-base-uncased \
    --epochs 3 \
    --batch_size 16 \
    --output_dir backend/models/bert_fake_reviews_model
```

**6. Analyze model errors**
```bash
python backend/modules/fake_reviews_error_analysis.py \
    --input data/processed/fake_reviews_cleaned.csv \
    --model_dir backend/models/bert_fake_reviews_model \
    --output_csv data/misclassified_reviews.csv
```

---

## Key findings

- AI-generated reviews in this dataset are systematically **shorter, less lexically diverse, and structurally simpler** than genuine reviews (73.6 vs. 61.3 words on average); star rating carries no discriminative signal.
- Combining TF-IDF with engineered linguistic features (**fusion model**) meaningfully outperforms either alone, confirming the two feature families capture complementary signal.
- The fine-tuned transformer's ~8-point accuracy gain over the best classical model is attributable to its sensitivity to GPT-2-specific generation artifacts - not a generic "deep learning is better" effect.
- Qualitative error analysis found the model's mistakes are systematic: it tends to flag unusually brief genuine reviews as fake, and is fooled by fake reviews that mimic genuine review conventions (structured pros/cons lists, disclosure language).

---

## References

- Salminen, J., Kandpal, C., Kamel, A. M., Jung, S., & Jansen, B. J. (2022). Creating and detecting fake reviews of online products. *Journal of Retailing and Consumer Services*, 64, 102771.
- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. *ACL-HLT 2011*.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.

---

## License / Academic Use

Built as an individual capstone project (DS4105) at Sabaragamuwa University of Sri Lanka, 2025/2026.