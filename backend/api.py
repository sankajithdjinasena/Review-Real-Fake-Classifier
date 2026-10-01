import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
MODULES_DIR = BACKEND_DIR / "modules"
MODELS_DIR = BACKEND_DIR / "models"
ROOT_DIR = BACKEND_DIR.parent

if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import fusion_inference
import scraper

app = FastAPI(
    title="Detecting Deception in Product Reviews - API",
    description="Backend API powering the React Product Review Classifier Dashboard",
    version="1.0.0",
)

# Enable CORS for development with Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Defaults
DEFAULT_MODEL_DIR = str(MODELS_DIR / "bert_fake_reviews_model")
DEFAULT_FUSION_DIR = str(MODELS_DIR / "fusion_model")

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
FAKE_CLASS_INDEX = 1
REAL_CLASS_INDEX = 0

# Cache for loaded models in memory
model_cache = {
    "bert_model": None,
    "bert_tokenizer": None,
    "bert_device": None,
    "bert_dir": None,
    "shap_explainer": None,
    "fusion_artifacts": None,
    "fusion_dir": None,
}


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return str(path)
    # Check in models dir
    if (MODELS_DIR / path_str).exists():
        return str(MODELS_DIR / path_str)
    if (MODELS_DIR / path.name).exists():
        return str(MODELS_DIR / path.name)
    # Check in backend dir
    if (BACKEND_DIR / path_str).exists():
        return str(BACKEND_DIR / path_str)
    # Check relative to working dir or root
    if (ROOT_DIR / path_str).exists():
        return str((ROOT_DIR / path_str).resolve())
    if path.exists():
        return str(path.resolve())
    return path_str



def get_bert(model_dir: str):
    model_dir = resolve_path(model_dir)
    if (
        model_cache["bert_model"] is not None
        and model_cache["bert_dir"] == model_dir
    ):
        return (
            model_cache["bert_model"],
            model_cache["bert_tokenizer"],
            model_cache["bert_device"],
        )

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Missing dependency torch/transformers: {e}"
        )

    if not os.path.isdir(model_dir):
        raise HTTPException(
            status_code=404, detail=f"Model directory not found: {model_dir}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Load the tokenizer saved with the trained DistilBERT model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load the trained classification model and move it to CPU or GPU
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(
            device
        )

        # Put the model into evaluation/inference mode
        model.eval()

        # Cache the model, tokenizer, device and model directory for reuse
        model_cache["bert_model"] = model
        model_cache["bert_tokenizer"] = tokenizer
        model_cache["bert_device"] = device
        model_cache["bert_dir"] = model_dir

        # Clear the old SHAP explainer because a different model may have been loaded
        model_cache["shap_explainer"] = None

        # Return everything needed for prediction
        return model, tokenizer, device

    except Exception as e:
        # Return an HTTP 500 error if the model cannot be loaded
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load DistilBERT model: {e}"
        )


def get_shap_explainer(model_dir: str):
    model_dir = resolve_path(model_dir)
    if (
        model_cache["shap_explainer"] is not None
        and model_cache["bert_dir"] == model_dir
    ):
        return model_cache["shap_explainer"]

    try:
        import shap
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail=f"Missing dependency shap: {e}"
        )

    model, tokenizer, device = get_bert(model_dir)

    def model_predict(texts):
        import torch

        texts = list(texts)
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items() if k != "token_type_ids"}
        with torch.no_grad():
            logits = model(**encoded).logits

            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.cpu().numpy()

    # Create a SHAP text masker using the tokenizer to determine which tokens can be hidden during explanation
    masker = shap.maskers.Text(tokenizer)

    # Create the SHAP explainer that tests masked versions of the text using the model's prediction function
    explainer = shap.Explainer(model_predict, masker)

    # Store the explainer in cache so it does not need to be created again for every request
    model_cache["shap_explainer"] = explainer

    # Return the initialized SHAP explainer
    return explainer


def get_fusion(fusion_dir: str):
    # Find the correct folder containing the saved Fusion Model files
    fusion_dir = resolve_path(fusion_dir)

    # If the Fusion Model was already loaded from this folder,
    # return it from memory instead of loading it again
    if (
        model_cache["fusion_artifacts"] is not None
        and model_cache["fusion_dir"] == fusion_dir
    ):
        return model_cache["fusion_artifacts"]

    # Load all saved Fusion Model components from the folder
    # e.g. TF-IDF vectorizer, scaler, Logistic Regression and SVM
    artifacts = fusion_inference.load_fusion_artifacts(fusion_dir)

    # If the required model files cannot be found, return a 404 error
    if artifacts is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Fusion model artifacts not found at: {fusion_dir}. "
                "Ensure tfidf_vectorizer.joblib, scaler.joblib, etc. exist."
            ),
        )

    # Save the loaded artifacts in memory so they can be reused
    model_cache["fusion_artifacts"] = artifacts
    model_cache["fusion_dir"] = fusion_dir

    # Return the loaded Fusion Model components
    return artifacts


from pydantic import BaseModel, ConfigDict

# Request format for predicting a manually entered review
class PredictRequest(BaseModel):
    # Allow field names such as model_choice/model_dir without Pydantic namespace warnings
    model_config = ConfigDict(protected_namespaces=())
    # Review text entered by the user
    text: str
    # Model selected for prediction; DistilBERT is the default
    model_choice: str = "DistilBERT (Transformer)"
    # Location of the saved DistilBERT model
    model_dir: Optional[str] = DEFAULT_MODEL_DIR
    # Location of the saved Fusion Model artifacts
    fusion_dir: Optional[str] = DEFAULT_FUSION_DIR


# Request format for scraping reviews from a product URL
class ScrapeRequest(BaseModel):
    # Allow field names such as model_choice/model_dir without Pydantic namespace warnings
    model_config = ConfigDict(protected_namespaces=())
    # Product URL from which reviews will be scraped
    url: str
    # Model used to classify the scraped reviews
    model_choice: str = "DistilBERT (Transformer)"
    # Maximum number of reviews to scrape; default is 12
    max_reviews: Optional[int] = 12
    # Location of the saved DistilBERT model
    model_dir: Optional[str] = DEFAULT_MODEL_DIR
    # Location of the saved Fusion Model artifacts
    fusion_dir: Optional[str] = DEFAULT_FUSION_DIR

# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/api/health")
def health_check():
    bert_path = resolve_path(DEFAULT_MODEL_DIR)
    fusion_path = resolve_path(DEFAULT_FUSION_DIR)
    return {
        "status": "online",
        "bert_model_available": os.path.isdir(bert_path),
        "fusion_model_available": os.path.isdir(fusion_path),
        "bert_path": bert_path,
        "fusion_path": fusion_path,
    }


@app.get("/api/stats")
def get_stats():
    return {
        "stats": [
            {"value": "40,405", "label": "reviews analyzed"},
            {"value": "10", "label": "product categories"},
            {"value": "4", "label": "modeling approaches tested"},
            {"value": "98.05%", "label": "best model accuracy"},
        ]
    }


@app.get("/api/dataset")
def get_dataset():
    return {
        "class_balance": [
            {"name": "Real", "value": 20215, "color": "#5FD3A0"},
            {"name": "Fake", "value": 20190, "color": "#E64980"},
        ],
        "review_length": [
            {"category": "avg. words per review", "Real": 73.6, "Fake": 61.3}
        ],
        "categories": [
            {"name": "Kindle Store", "count": 4700},
            {"name": "Books", "count": 4400},
            {"name": "Pet Supplies", "count": 4250},
            {"name": "Home & Kitchen", "count": 4050},
            {"name": "Electronics", "count": 4000},
            {"name": "Sports & Outdoors", "count": 3950},
            {"name": "Tools & Home Impr.", "count": 3900},
            {"name": "Clothing/Shoes/Jewelry", "count": 3850},
            {"name": "Toys & Games", "count": 3800},
            {"name": "Movies & TV", "count": 3600},
        ],
        "rating_distribution": [
            {"rating": "1", "Fake": 1050, "Real": 1050},
            {"rating": "2", "Fake": 950, "Real": 950},
            {"rating": "3", "Fake": 1950, "Real": 1850},
            {"rating": "4", "Fake": 3950, "Real": 4050},
            {"rating": "5", "Fake": 12250, "Real": 12250},
        ],
    }


@app.get("/api/investigation")
def get_investigation():
    return {
        "models": [
            {
                "name": "Engineered Features (Random Forest)",
                "accuracy": 85.27,
                "color": "#8DA0A8",
            },
            {
                "name": "TF-IDF (Linear SVM)",
                "accuracy": 87.82,
                "color": "#7BB6C9",
            },
            {
                "name": "Fusion Model (TF-IDF + Engineered)",
                "accuracy": 90.89,
                "color": "#F4C95D",
            },
            {
                "name": "DistilBERT (fine-tuned)",
                "accuracy": 98.05,
                "color": "#5FD3A0",
            },
        ]
    }


@app.get("/api/verdict")
def get_verdict():
    return {
        "confusion_matrix": [
            {"y": "Actual Real", "Predicted Real": 2941, "Predicted Fake": 92},
            {"y": "Actual Fake", "Predicted Real": 26, "Predicted Fake": 3002},
        ],
        "metrics": {
            "accuracy": "98.05%",
            "f1": "98.07%",
            "precision": "97.03%",
            "recall": "99.14%",
        },
        "sample_size": "6,061 held-out reviews",
    }


@app.get("/api/case-notes")
def get_case_notes():
    return {
        "false_positives": [
            {
                "quote": (
                    "I enjoyed the two stories contained in this book. "
                    "These are the only books by these authors that I have read."
                ),
                "note": (
                    "Short, blunt, factual - no personal elaboration. The model appears to associate "
                    "brevity and genericness with AI generation, penalizing naturally terse human writers."
                ),
            },
            {
                "quote": (
                    "They fit perfect, they look expensive, they are the most comfortable shoes "
                    "that i had ever. I love the design."
                ),
                "note": (
                    "Enthusiastic but generic praise, close to the kind of language GPT-2 tends to generate."
                ),
            },
            {
                "quote": (
                    "I have two of these, one for the kitchen and one for the dining room."
                ),
                "note": (
                    "Minimal, matter-of-fact - the shortest kind of real review, easily confused "
                    "with generic filler text."
                ),
            },
            {
                "quote": (
                    "Not at all what I was expecting. This was a large box of plastic pieces that "
                    "don't fit together. I'm sure the only way this toy would be usable is if you "
                    "glued the pieces together."
                ),
                "note": (
                    "A genuine complaint, but structured plainly enough to read as generic negative filler."
                ),
            },
            {
                "quote": (
                    "Very comfortable, just a little short for my taste. I also have long legs, "
                    "so that might be the issue."
                ),
                "note": (
                    "Specific personal detail (long legs) still wasn't enough to overcome the "
                    "short-review heuristic."
                ),
            },
        ],
        "false_negatives": [
            {
                "quote": (
                    "We used these with 3 of my 3 year old grandkids. They really enjoy playing with them."
                ),
                "note": (
                    "Specific, human-sounding family detail - exactly the kind of anecdotal texture "
                    "real reviews have."
                ),
            },
            {
                "quote": (
                    "Pros: Wash well. Laundry bag included, thicker than expected, great quality "
                    "materials. Cons: Too small for my petite frame, I had to return it."
                ),
                "note": (
                    "Mimics a genuine human review convention - structured pros/cons lists are common "
                    "in real reviews, so the model reads structure as authenticity."
                ),
            },
            {
                "quote": (
                    "Received product at discount for honest review. Waking Up by Kirsten Clare. "
                    "I am a huge fan of both Haus of Tars and The Blind Side."
                ),
                "note": (
                    "Borrows a real, common Amazon disclosure phrase - a genuine trust signal the "
                    "fake text co-opts convincingly."
                ),
            },
            {
                "quote": (
                    "Contains raw embedded HTML markup (a product link tag) rather than natural review prose."
                ),
                "note": (
                    "Likely not a genuine model failure - this looks like a data leakage or scraping "
                    "artifact mislabeled as 'fake' in the source dataset."
                ),
            },
            {
                "quote": "SHEET COLOR IS NICE BUY FOR MY SIZE FOR MY TOWN.",
                "note": (
                    "Ungrammatical, broken English - the model likely treats imperfect fluency as a "
                    "human tell, which a fake review can exploit."
                ),
            },
        ],
    }
@app.post("/api/predict")
def predict_review(req: PredictRequest):
    # Get review text and remove leading/trailing spaces
    text = req.text.strip()

    # Reject empty reviews or reviews with fewer than 3 words
    if not text or len(text.split()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Review text must be at least 3 words long.",
        )

    # Check whether a Fusion model is selected
    is_fusion = req.model_choice.startswith("Fusion")

    # Select Logistic Regression or Linear SVM for Fusion prediction
    fusion_which = "logreg" if "Logistic" in req.model_choice else "svm"

    if is_fusion:
        # Load cached/saved Fusion model components
        fusion_dir = req.fusion_dir or DEFAULT_FUSION_DIR
        artifacts = get_fusion(fusion_dir)

        # Predict REAL/FAKE using the selected Fusion classifier
        result = fusion_inference.predict_fusion(
            text, artifacts, which=fusion_which
        )

        # Calculate feature contributions for the Fusion prediction
        explain_df = fusion_inference.explain_fusion(
            text, artifacts, which=fusion_which
        )

        contributions = []
        top_influential = []

        if not explain_df.empty:
            # Positive values push toward FAKE
            pos = (
                explain_df[explain_df["shap_value"] > 0]
                .sort_values("shap_value", ascending=False)
                .head(10)
            )

            # Negative values push toward REAL
            neg = (
                explain_df[explain_df["shap_value"] < 0]
                .sort_values("shap_value", ascending=True)
                .head(10)
            )

            # Combine REAL-direction and FAKE-direction contributions
            subset = (
                import_pandas_concat([neg, pos])
                if not (pos.empty and neg.empty)
                else explain_df
            )
            subset = subset.sort_values("shap_value")

            # Prepare feature contributions for the frontend
            for _, row in subset.iterrows():
                val = float(row["shap_value"])
                contributions.append(
                    {
                        "display_token": str(row["display_token"]),
                        "shap_value": round(val, 4),
                        "direction": "FAKE" if val > 0 else "REAL",
                    }
                )

            # Find the 10 strongest features regardless of direction
            top_df = (
                explain_df.assign(abs_shap=lambda d: d["shap_value"].abs())
                .sort_values("abs_shap", ascending=False)
                .head(10)
            )

            for _, row in top_df.iterrows():
                val = float(row["shap_value"])
                top_influential.append(
                    {
                        "token": str(row["display_token"]),
                        "contribution": round(val, 4),
                        "pushes_toward": "→ FAKE" if val > 0 else "→ REAL",
                    }
                )

        # Add explanation note for calibrated Linear SVM
        note = (
            "Linear SVM values are averaged across the model's "
            "calibration folds - a close approximation."
            if fusion_which == "svm"
            else None
        )

        # Return Fusion prediction to React as JSON
        return {
            "label": result["label"],
            "confidence": round(result["confidence"], 6),
            "prob_real": round(result["prob_real"], 6),
            "prob_fake": round(result["prob_fake"], 6),
            "contributions": contributions,
            "top_influential": top_influential,
            "method_note": note,
            "device": "CPU (Scikit-Learn)",
        }

    # Load cached/saved DistilBERT model and tokenizer
    model_dir = req.model_dir or DEFAULT_MODEL_DIR
    model, tokenizer, device = get_bert(model_dir)

    import torch

    # Map model class IDs to readable labels
    id2label = {
        REAL_CLASS_INDEX: "real",
        FAKE_CLASS_INDEX: "fake",
    }

    # Run prediction without calculating gradients
    with torch.no_grad():
        # Convert review text into DistilBERT input tokens
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(device)

        # DistilBERT does not require token_type_ids
        enc = {k: v for k, v in enc.items() if k != "token_type_ids"}

        # Pass tokens through DistilBERT and get raw scores
        logits = model(**enc).logits

        # Convert logits into REAL/FAKE probabilities
        probs = torch.softmax(logits, dim=-1)[0]

        # Select the class with the highest probability
        pred_id = int(torch.argmax(probs).item())

    # Extract class probabilities and final prediction
    prob_real = float(probs[REAL_CLASS_INDEX].item())
    prob_fake = float(probs[FAKE_CLASS_INDEX].item())
    label = id2label[pred_id]
    confidence = float(probs[pred_id].item())

    contributions = []
    top_influential = []

    # Generate token-level SHAP explanation
    try:
        # Load/create cached SHAP explainer
        explainer = get_shap_explainer(model_dir)

        # Mask different tokens and measure their influence on prediction
        shap_values = explainer([text])

        # Get each token's contribution toward the FAKE class
        values = shap_values.values[0, :, FAKE_CLASS_INDEX]
        tokens = shap_values.data[0]

        raw_rows = []

        # Match each valid token with its SHAP value
        for tok, val in zip(tokens, values):
            tok = str(tok)

            # Ignore empty and special tokens
            if not tok.strip() or tok in SPECIAL_TOKENS:
                continue

            raw_rows.append(
                {
                    "token": tok,
                    "shap_value": float(val),
                }
            )

        if raw_rows:
            merged = []

            # Merge WordPiece sub-tokens into readable words
            # Example: "un" + "##believable" → "unbelievable"
            for row in raw_rows:
                tok = row["token"]
                val = row["shap_value"]

                if tok.startswith("##") and merged:
                    merged[-1]["display_token"] += tok[2:]
                    merged[-1]["shap_value"] += val
                else:
                    merged.append(
                        {
                            "display_token": tok.strip(),
                            "shap_value": val,
                        }
                    )

            import pandas as pd

            # Convert token contributions into a DataFrame
            shap_df = pd.DataFrame(merged)

            # Get 10 strongest tokens pushing toward FAKE
            pos = (
                shap_df[shap_df["shap_value"] > 0]
                .sort_values("shap_value", ascending=False)
                .head(10)
            )

            # Get 10 strongest tokens pushing toward REAL
            neg = (
                shap_df[shap_df["shap_value"] < 0]
                .sort_values("shap_value", ascending=True)
                .head(10)
            )

            # Combine REAL and FAKE contributions
            subset = pd.concat([neg, pos]).sort_values("shap_value")

            # Prepare SHAP contributions for the frontend
            for _, row in subset.iterrows():
                val = float(row["shap_value"])
                contributions.append(
                    {
                        "display_token": str(row["display_token"]),
                        "shap_value": round(val, 4),
                        "direction": "FAKE" if val > 0 else "REAL",
                    }
                )

            # Find the 10 most influential tokens regardless of direction
            top_df = (
                shap_df.assign(abs_shap=lambda d: d["shap_value"].abs())
                .sort_values("abs_shap", ascending=False)
                .head(10)
            )

            for _, row in top_df.iterrows():
                val = float(row["shap_value"])
                top_influential.append(
                    {
                        "token": str(row["display_token"]),
                        "contribution": round(val, 4),
                        "pushes_toward": "→ FAKE" if val > 0 else "→ REAL",
                    }
                )

    # Prediction still works if SHAP explanation fails
    except Exception as e:
        print(f"SHAP extraction note: {e}")

    # Return DistilBERT prediction and SHAP explanation to React
    return {
        "label": label,
        "confidence": round(confidence, 6),
        "prob_real": round(prob_real, 6),
        "prob_fake": round(prob_fake, 6),
        "contributions": contributions,
        "top_influential": top_influential,
        "method_note": (
            "SHAP feature contribution values over DistilBERT transformer tokens."
        ),
        "device": device.upper(),
    }


def import_pandas_concat(objs):
    # Combine multiple pandas DataFrames into one
    import pandas as pd
    return pd.concat(objs)


@app.post("/api/scrape-and-predict")
def scrape_and_predict(req: ScrapeRequest):
    # Get product URL and remove extra spaces
    url = req.url.strip()

    # Reject an empty URL
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    try:
        # Scrape product information and reviews from the URL
        scraped_data = scraper.scrape_reviews_from_url(
            url,
            max_reviews=req.max_reviews or 12,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to scrape URL: {e}",
        )

    # Extract scraped reviews
    scraped_reviews = scraped_data["reviews"]

    # Return an error if no reviews were found
    if not scraped_reviews:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any review text from the specified URL.",
        )

    results = []
    real_count = 0
    fake_count = 0

    # Predict each scraped review individually
    for item in scraped_reviews:
        # Create a prediction request using the selected model
        predict_req = PredictRequest(
            text=item["text"],
            model_choice=req.model_choice,
            model_dir=req.model_dir,
            fusion_dir=req.fusion_dir,
        )

        # Reuse the existing prediction function
        res = predict_review(predict_req)

        # Add original review information to the result
        res["title"] = item["title"]
        res["rating"] = item["rating"]
        res["text"] = item["text"]

        # Count REAL and FAKE predictions
        if res["label"] == "real":
            real_count += 1
        else:
            fake_count += 1

        results.append(res)

    # Calculate total analyzed reviews
    total = len(results)

    # Trust score = percentage of analyzed reviews predicted as REAL
    trust_score = round((real_count / total) * 100, 1) if total > 0 else 0.0

    # Return product-level and individual review results to React
    return {
        "product_title": scraped_data["product_title"],
        "url": url,
        "trust_score": trust_score,
        "total_reviews": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "reviews": results,
    }


# Locate the production React/Vite build folder
DIST_DIR = ROOT_DIR / "frontend" / "dist"

# Serve the built React frontend through FastAPI if it exists
if DIST_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(DIST_DIR), html=True),
        name="static",
    )


# Start the FastAPI server only when this file is executed directly
if __name__ == "__main__":
    import uvicorn

    # Run backend on localhost:8000 and reload after code changes
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )