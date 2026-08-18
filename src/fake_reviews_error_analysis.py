import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RANDOM_STATE = 42
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}


def load_test_split(path: str):
    """
    Recreates the EXACT same train/val/test split used during fine-tuning
    (same random_state, same split ratios), so the test set here is
    identical to what the model was evaluated on originally.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["review_text", "label_clean"]).reset_index(drop=True)
    df["label_id"] = df["label_clean"].map(LABEL2ID)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label_id"]
    )
    print(f"Recreated test set: {len(test_df)} rows (should match your training run's test size)")
    return test_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Extract misclassified reviews from the fine-tuned model.")
    parser.add_argument("--input", required=True, help="Path to the cleaned dataset CSV.")
    parser.add_argument("--model_dir", required=True, help="Path to the saved fine-tuned model directory.")
    parser.add_argument("--output_csv", default="misclassified_reviews.csv")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_df = load_test_split(args.input)

    print(f"Loading fine-tuned model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    texts = test_df["review_text"].tolist()
    true_labels = test_df["label_id"].tolist()

    all_preds = []
    all_confidences = []

    print("Running inference on test set...")
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            enc = tokenizer(
                batch_texts, truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values

            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

            if (i // args.batch_size) % 20 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)}")

    test_df["true_label"] = [ID2LABEL[l] for l in true_labels]
    test_df["predicted_label"] = [ID2LABEL[p] for p in all_preds]
    test_df["model_confidence"] = all_confidences
    test_df["correct"] = test_df["true_label"] == test_df["predicted_label"]

    misclassified = test_df[~test_df["correct"]].copy()
    misclassified = misclassified.sort_values("model_confidence", ascending=False)

    print(f"\nTotal test set: {len(test_df)}")
    print(f"Misclassified: {len(misclassified)}")

    false_positives = misclassified[
        (misclassified["true_label"] == "real") & (misclassified["predicted_label"] == "fake")
    ]
    false_negatives = misclassified[
        (misclassified["true_label"] == "fake") & (misclassified["predicted_label"] == "real")
    ]
    print(f"False positives (real flagged as fake): {len(false_positives)}")
    print(f"False negatives (fake missed as real):  {len(false_negatives)}")

    cols_to_save = ["review_text", "true_label", "predicted_label", "model_confidence", "rating", "category"]
    cols_to_save = [c for c in cols_to_save if c in misclassified.columns]
    misclassified[cols_to_save].to_csv(args.output_csv, index=False)
    print(f"\nAll misclassified reviews saved to: {args.output_csv}")

    # Print the highest-confidence mistakes — these are the most interesting
    # ones for discussion, since the model was CONFIDENTLY wrong.
    print("\n" + "=" * 70)
    print("  TOP 5 MOST CONFIDENT FALSE POSITIVES (real review flagged as fake)")
    print("=" * 70)
    for _, row in false_positives.head(5).iterrows():
        print(f"\n[confidence={row['model_confidence']:.3f}] {row['review_text'][:300]}")

    print("\n" + "=" * 70)
    print("  TOP 5 MOST CONFIDENT FALSE NEGATIVES (fake review missed as real)")
    print("=" * 70)
    for _, row in false_negatives.head(5).iterrows():
        print(f"\n[confidence={row['model_confidence']:.3f}] {row['review_text'][:300]}")

    print(
        "\nFor your report: read through misclassified_reviews.csv and look for "
        "patterns — e.g. are false negatives unusually long/detailed fake reviews "
        "that 'read' as human? Are false positives short, blunt real reviews that "
        "resemble generic AI text? Quoting 2-3 representative examples (paraphrased, "
        "not verbatim if publishing) with your interpretation is strong material for "
        "Chapter 5's 'Limitations and potential biases' subsection."
    )


if __name__ == "__main__":
    main()
