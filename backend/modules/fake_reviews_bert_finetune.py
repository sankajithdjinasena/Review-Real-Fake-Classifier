# CELL 1 - Import libraries for data handling, PyTorch, evaluation and transformer training
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
# AutoTokenizer → converts text into tokens
# AutoModelForSequenceClassification → DistilBERT for classification
# TrainingArguments → training configuration
# Trainer → handles training
# EarlyStoppingCallback → stops training when validation performance stops improving

# CELL 2 - Set reproducible random seed and map class names to numerical labels
RANDOM_STATE = 42
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

# CELL 3 - Load dataset, remove missing values and create stratified 70/15/15 split
def load_and_split(path, sample):
    df = pd.read_csv(path)
    df = df.dropna(subset=["review_text", "label_clean"]).reset_index(drop=True)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows for a quick test run.")
    df["label_id"] = df["label_clean"].map(LABEL2ID)
    # 70% train / 15% val / 15% test, stratified → This keeps the class distribution approximately the same in each split.
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label_id"]
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df

# CELL 4 - Convert reviews into fixed-length transformer tokens and PyTorch tensors
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", # truncation - If a review is longer than 256 tokens it gets shortened to 256 tokens. Padding - If a review is shorter than 256 tokens it gets padded with zeros to reach 256 tokens.
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# CELL 5 - Convert each dataframe split into a tokenized ReviewDataset
def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int = 256):
    return ReviewDataset(df["review_text"], df["label_id"], tokenizer, max_length=max_length)

# CELL 6 - Convert model logits into predictions and calculate classification metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Separate model prediction scores (logits) and actual labels
    preds = np.argmax(logits, axis=-1)  # Select the class with the highest score as the predicted label
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_fake": precision_score(labels, preds, pos_label=1),
        "recall_fake": recall_score(labels, preds, pos_label=1),
        "f1_fake": f1_score(labels, preds, pos_label=1),
    }

# CELL 7 - Configure command-line arguments for dataset, model and training settings
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for fake review detection.")
    parser.add_argument("--input", required=True, help="Path to cleaned dataset CSV (needs review_text, label_clean).")
    parser.add_argument("--model_name", default="distilbert-base-uncased",
                         help="HF model name: distilbert-base-uncased | bert-base-uncased | roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", default="./bert_fake_reviews_model")
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: run on a subset first to validate the pipeline before the full run.")
    args = parser.parse_args()

    # CELL 8 - Detect GPU/CPU and prepare train, validation and test datasets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Fine-tuning on CPU will be very slow - "
              "use --sample for a small test, or switch to a GPU runtime (Colab: Runtime > Change runtime type > GPU).")
    train_df, val_df, test_df = load_and_split(args.input, sample=args.sample)

    # CELL 9 - Load pretrained tokenizer and transformer with binary classification output
    print(f"\nLoading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,        # Load the selected pretrained transformer model, e.g. DistilBERT
    num_labels=2,           # Two output classes: real and fake
    id2label=ID2LABEL,      # Convert predicted IDs to names: 0 → real, 1 → fake
    label2id=LABEL2ID       # Convert label names to IDs: real → 0, fake → 1
)

    # CELL 10 - Tokenize training, validation and test reviews
    print("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_df, tokenizer, args.max_length)
    val_ds = tokenize_dataset(val_df, tokenizer, args.max_length)
    test_ds = tokenize_dataset(test_df, tokenizer, args.max_length)

    # CELL 11 - Configure epochs, batch sizes, evaluation, checkpoints and mixed precision
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, # How many reviews are processed together during training.
        per_device_eval_batch_size=args.batch_size, # How many reviews are processed together during evaluation.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake", # Choose the model with the highest fake-review F1.
        greater_is_better=True,
        logging_steps=50,  # Log training information such as loss every 50 training steps
        report_to="none",  # Do not send training logs to external tools such as Weights & Biases
        fp16=torch.cuda.is_available(),  # Use 16-bit floating-point training when CUDA GPU is available; reduces memory use and usually speeds up training
        save_total_limit=2,  # Keep only 2 saved checkpoints to prevent excessive disk usage
    )

    # CELL 12 - Create Trainer and apply early stopping using validation performance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # CELL 13 - Fine-tune the pretrained transformer on the training dataset
    print("\nStarting fine-tuning...")
    trainer.train()

    # CELL 14 - Evaluate the best checkpoint on the unseen held-out test dataset
    print("\nEvaluating on held-out TEST set...")
    test_results = trainer.evaluate(test_ds)
    print("\nTest set results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # CELL 15 - Generate test predictions and calculate confusion matrix
    # Full confusion matrix on test set for the report
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print("\nConfusion matrix [rows=actual, cols=predicted] order=[real, fake]:")
    print(cm)

    # CELL 16 - Save the final trained model and matching tokenizer
    print(f"\nSaving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # CELL 17 - Compare transformer results against the strongest classical fusion model
    print(
        "\nCompare test_results above against your classical fusion model "
        "(90.89% accuracy, 0.908 F1). Report both, and discuss the gap "
        "(if any) in terms of whether the added compute cost and reduced "
        "interpretability of the transformer is justified by the performance gain."
    )

# CELL 18 - Execute the complete pipeline when this Python script is run directly
if __name__ == "__main__":
    main()