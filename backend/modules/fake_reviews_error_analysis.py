# CELL 1 - Import libraries for command-line arguments, data handling, PyTorch and transformer inference
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CELL 2 - Use the same random seed and label mapping as the training script
RANDOM_STATE = 42
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

# CELL 3 - Recreate exactly the same 15% held-out test split used during model training
def load_test_split(path: str):
    df = pd.read_csv(path)  # Load cleaned review dataset
    df = df.dropna(subset=["review_text", "label_clean"]).reset_index(drop=True)  # Remove missing reviews/labels
    df["label_id"] = df["label_clean"].map(LABEL2ID)  # Convert real/fake labels into 0/1
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["label_id"]
    )  # 70% train and 30% temporary set while maintaining class distribution
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label_id"]
    )  # Split temporary data equally into 15% validation and 15% test
    print(f"Recreated test set: {len(test_df)} rows (should match your training run's test size)")
    return test_df.reset_index(drop=True)  # Return only the held-out test set

# CELL 4 - Read command-line settings for dataset, saved model, output file and inference configuration
def main():
    parser = argparse.ArgumentParser(description="Extract misclassified reviews from the fine-tuned model.")
    parser.add_argument("--input", required=True, help="Path to the cleaned dataset CSV.")
    parser.add_argument("--model_dir", required=True, help="Path to the saved fine-tuned model directory.")
    parser.add_argument("--output_csv", default="misclassified_reviews.csv")
    parser.add_argument("--max_length", type=int, default=256)  # Maximum number of tokens per review
    parser.add_argument("--batch_size", type=int, default=32)  # Number of reviews processed together
    args = parser.parse_args()

    # CELL 5 - Select GPU if CUDA is available; otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CELL 6 - Recreate the held-out test dataset used during training
    test_df = load_test_split(args.input)

    # CELL 7 - Load the saved tokenizer and fine-tuned model and switch model to evaluation mode
    print(f"Loading fine-tuned model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)  # Load tokenizer saved with trained model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)  # Load model and move to GPU/CPU
    model.eval()  # Disable training-specific behaviour such as dropout

    # CELL 8 - Extract review texts and their true labels from the test set
    texts = test_df["review_text"].tolist()
    true_labels = test_df["label_id"].tolist()
    all_preds = []  # Store predicted class for every review
    all_confidences = []  # Store confidence of each prediction

    # CELL 9 - Run model inference on the test reviews in batches
    print("Running inference on test set...")
    with torch.no_grad():  # Disable gradient calculation because no training is performed
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]  # Select current batch of reviews
            enc = tokenizer(
                batch_texts, truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt"
            ).to(device)  # Tokenize reviews and move tensors to GPU/CPU
            logits = model(**enc).logits  # Get raw output scores for real and fake classes
            probs = torch.softmax(logits, dim=-1)  # Convert logits into probabilities that sum to 1
            preds = torch.argmax(probs, dim=-1)  # Select class with highest probability
            confidences = torch.max(probs, dim=-1).values  # Get probability of selected class
            all_preds.extend(preds.cpu().tolist())  # Move predictions to CPU and store them
            all_confidences.extend(confidences.cpu().tolist())  # Store prediction confidence
            if (i // args.batch_size) % 20 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)}")

    # CELL 10 - Add true labels, predictions, confidence and correctness to the test dataframe
    test_df["true_label"] = [ID2LABEL[l] for l in true_labels]  # Convert 0/1 true labels back to real/fake
    test_df["predicted_label"] = [ID2LABEL[p] for p in all_preds]  # Convert predictions back to real/fake
    test_df["model_confidence"] = all_confidences  # Add prediction confidence
    test_df["correct"] = test_df["true_label"] == test_df["predicted_label"]  # Check whether prediction is correct

    # CELL 11 - Extract only incorrect predictions and rank them by model confidence
    misclassified = test_df[~test_df["correct"]].copy()  # Keep rows where prediction != true label
    misclassified = misclassified.sort_values("model_confidence", ascending=False)  # Highest-confidence mistakes first
    print(f"\nTotal test set: {len(test_df)}")
    print(f"Misclassified: {len(misclassified)}")

    # CELL 12 - Separate mistakes into false positives and false negatives
    false_positives = misclassified[
        (misclassified["true_label"] == "real") & (misclassified["predicted_label"] == "fake")
    ]  # Genuine review incorrectly classified as fake
    false_negatives = misclassified[
        (misclassified["true_label"] == "fake") & (misclassified["predicted_label"] == "real")
    ]  # Fake review incorrectly classified as real
    print(f"False positives (real flagged as fake): {len(false_positives)}")
    print(f"False negatives (fake missed as real):  {len(false_negatives)}")

    # CELL 13 - Save misclassified reviews and useful information into a CSV file
    cols_to_save = ["review_text", "true_label", "predicted_label", "model_confidence", "rating", "category"]
    cols_to_save = [c for c in cols_to_save if c in misclassified.columns]  # Keep only columns that exist
    misclassified[cols_to_save].to_csv(args.output_csv, index=False)
    print(f"\nAll misclassified reviews saved to: {args.output_csv}")

    # CELL 14 - Display the five highest-confidence false positives for error analysis
    # Print the highest-confidence mistakes - these are the most interesting
    # ones for discussion, since the model was CONFIDENTLY wrong.
    print("\n" + "=" * 70)
    print("  TOP 5 MOST CONFIDENT FALSE POSITIVES (real review flagged as fake)")
    print("=" * 70)
    for _, row in false_positives.head(5).iterrows():
        print(f"\n[confidence={row['model_confidence']:.3f}] {row['review_text'][:300]}")

    # CELL 15 - Display the five highest-confidence false negatives for error analysis
    print("\n" + "=" * 70)
    print("  TOP 5 MOST CONFIDENT FALSE NEGATIVES (fake review missed as real)")
    print("=" * 70)
    for _, row in false_negatives.head(5).iterrows():
        print(f"\n[confidence={row['model_confidence']:.3f}] {row['review_text'][:300]}")

# CELL 16 - Execute the misclassification analysis pipeline when the script is run directly
if __name__ == "__main__":
    main()