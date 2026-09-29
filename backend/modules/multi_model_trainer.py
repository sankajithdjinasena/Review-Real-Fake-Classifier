
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

RANDOM_STATE = 42
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}

MODEL_PRESETS = {
    "distilbert": "distilbert-base-uncased",
    "deberta-v3": "microsoft/deberta-v3-small",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "electra": "google/electra-base-discriminator",
    "mobilebert": "google/mobilebert-uncased",
    "longformer": "allenai/longformer-base-4096",
}


# ==========================================
# Data Splitting Function
# ==========================================
def load_and_split_data(csv_path, sample=None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["review_text", "label_clean"]).reset_index(drop=True)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled dataset down to {len(df)} rows.")

    df["label"] = df["label_clean"].map(LABEL2ID)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_STATE, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label"]
    )

    print(f"Data Split -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ==========================================
# Evaluation Helper Function
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_fake": precision_score(labels, preds, pos_label=1),
        "recall_fake": recall_score(labels, preds, pos_label=1),
        "f1_fake": f1_score(labels, preds, pos_label=1),
    }


# ==========================================
# Procedural Transformer Trainer (Pre-built HF functions)
# ==========================================
def train_transformer_model(model_key, model_name, train_df, val_df, test_df, args):
    print(f"\n[{model_key.upper()}] Loading Model & Tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Use built-in Hugging Face Dataset.from_pandas (No custom class required!)
    def tokenize_function(batch):
        return tokenizer(batch["review_text"], truncation=True, padding="max_length", max_length=args.max_length)

    train_ds = HFDataset.from_pandas(train_df[["review_text", "label"]]).map(tokenize_function, batched=True)
    val_ds = HFDataset.from_pandas(val_df[["review_text", "label"]]).map(tokenize_function, batched=True)
    test_ds = HFDataset.from_pandas(test_df[["review_text", "label"]]).map(tokenize_function, batched=True)

    out_dir = os.path.join(args.output_dir, model_key)

    use_fp16 = torch.cuda.is_available() and ("deberta" not in model_key.lower())
    use_bf16 = torch.cuda.is_available() and ("deberta" in model_key.lower()) and torch.cuda.is_bf16_supported()

    # Optimal learning rates per model family to prevent collapse (DeBERTa/MobileBERT need 2e-5 + warmup)
    lr = args.learning_rate if args.learning_rate is not None else (2e-5 if ("deberta" in model_key.lower() or "mobilebert" in model_key.lower()) else 3e-5)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        save_total_limit=1,
    )

    # Pre-built Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    eval_start = time.time()
    test_results = trainer.evaluate(test_ds)
    eval_time = time.time() - eval_start

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    return {
        "model": model_key,
        "hf_name": model_name,
        "accuracy": test_results.get("eval_accuracy", 0),
        "precision_fake": test_results.get("eval_precision_fake", 0),
        "recall_fake": test_results.get("eval_recall_fake", 0),
        "f1_fake": test_results.get("eval_f1_fake", 0),
        "train_time_sec": round(train_time, 2),
        "test_latency_sec": round(eval_time, 2),
    }, model


# ==========================================
# Procedural Knowledge Distillation (No custom Trainer class!)
# ==========================================
def train_distillation_procedural(teacher_model_dir, train_df, val_df, test_df, args):
    print("\n[KNOWLEDGE DISTILLATION] Running Procedural Distillation (Teacher -> Student)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    student_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(student_name)

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_model_dir).to(device)
    student = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=2).to(device)
    teacher.eval()

    def tokenize_fn(batch):
        return tokenizer(batch["review_text"], truncation=True, padding="max_length", max_length=args.max_length)

    train_ds = HFDataset.from_pandas(train_df[["review_text", "label"]]).map(tokenize_fn, batched=True)
    val_ds = HFDataset.from_pandas(val_df[["review_text", "label"]]).map(tokenize_fn, batched=True)
    test_ds = HFDataset.from_pandas(test_df[["review_text", "label"]]).map(tokenize_fn, batched=True)

    # Convert to standard PyTorch DataLoaders using TensorDataset (No custom class!)
    def build_loader(ds, batch_size, shuffle=False):
        input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(ds["attention_mask"], dtype=torch.long)
        labels = torch.tensor(ds["label"], dtype=torch.long)
        tds = TensorDataset(input_ids, attention_mask, labels)
        return DataLoader(tds, batch_size=batch_size, shuffle=shuffle)

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = build_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
    temperature = 2.0
    alpha = 0.5

    start_time = time.time()
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        for b_ids, b_mask, b_labels in train_loader:
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

            optimizer.zero_grad()
            s_out = student(input_ids=b_ids, attention_mask=b_mask)
            with torch.no_grad():
                t_out = teacher(input_ids=b_ids, attention_mask=b_mask)

            loss_kd = F.kl_div(
                F.log_softmax(s_out.logits / temperature, dim=-1),
                F.softmax(t_out.logits / temperature, dim=-1),
                reduction="batchmean"
            ) * (temperature ** 2)

            loss_ce = F.cross_entropy(s_out.logits, b_labels)
            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Distillation Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")

    train_time = time.time() - start_time

    # Test Evaluation
    student.eval()
    eval_start = time.time()
    preds, targets = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_labels in test_loader:
            b_ids, b_mask = b_ids.to(device), b_mask.to(device)
            out = student(input_ids=b_ids, attention_mask=b_mask)
            p = torch.argmax(out.logits, dim=-1)
            preds.extend(p.cpu().numpy())
            targets.extend(b_labels.numpy())

    eval_time = time.time() - eval_start

    out_dir = os.path.join(args.output_dir, "knowledge_distillation_student")
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    return {
        "model": "knowledge_distillation",
        "hf_name": f"Teacher({teacher_model_dir})->Student(DistilBERT)",
        "accuracy": accuracy_score(targets, preds),
        "precision_fake": precision_score(targets, preds, pos_label=1),
        "recall_fake": recall_score(targets, preds, pos_label=1),
        "f1_fake": f1_score(targets, preds, pos_label=1),
        "train_time_sec": round(train_time, 2),
        "test_latency_sec": round(eval_time, 2),
    }


# ==========================================
# Procedural BiLSTM (Using pre-built nn.Sequential - No custom class!)
# ==========================================
def train_bilstm_procedural(train_df, val_df, test_df, args):
    print("\n[BiLSTM] Training Procedural BiLSTM using PyTorch built-in modules")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["review_text"], truncation=True, padding="max_length", max_length=args.max_length)

    train_ds = HFDataset.from_pandas(train_df[["review_text", "label"]]).map(tokenize_fn, batched=True)
    test_ds = HFDataset.from_pandas(test_df[["review_text", "label"]]).map(tokenize_fn, batched=True)

    def build_loader(ds, batch_size, shuffle=False):
        input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
        labels = torch.tensor(ds["label"], dtype=torch.long)
        tds = TensorDataset(input_ids, labels)
        return DataLoader(tds, batch_size=batch_size, shuffle=shuffle)

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False)

    # Pre-built PyTorch Layer Objects (No custom class required!)
    embedding_layer = nn.Embedding(tokenizer.vocab_size, 128, padding_idx=0).to(device)
    lstm_layer = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3).to(device)
    fc_layer = nn.Linear(256, 2).to(device)

    optimizer = torch.optim.AdamW(
        list(embedding_layer.parameters()) + list(lstm_layer.parameters()) + list(fc_layer.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(args.epochs):
        embedding_layer.train(); lstm_layer.train(); fc_layer.train()
        total_loss = 0
        for b_ids, b_labels in train_loader:
            b_ids, b_labels = b_ids.to(device), b_labels.to(device)

            optimizer.zero_grad()
            x = embedding_layer(b_ids)
            lstm_out, (ht, _) = lstm_layer(x)
            hidden = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
            logits = fc_layer(hidden)

            loss = criterion(logits, b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"BiLSTM Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")

    train_time = time.time() - start_time

    # Test Evaluation
    embedding_layer.eval(); lstm_layer.eval(); fc_layer.eval()
    eval_start = time.time()
    preds, targets = [], []
    with torch.no_grad():
        for b_ids, b_labels in test_loader:
            b_ids = b_ids.to(device)
            x = embedding_layer(b_ids)
            _, (ht, _) = lstm_layer(x)
            hidden = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
            logits = fc_layer(hidden)
            p = torch.argmax(logits, dim=-1)
            preds.extend(p.cpu().numpy())
            targets.extend(b_labels.numpy())

    eval_time = time.time() - eval_start

    return {
        "model": "bilstm",
        "hf_name": "BiLSTM (Procedural PyTorch Sequential)",
        "accuracy": accuracy_score(targets, preds),
        "precision_fake": precision_score(targets, preds, pos_label=1),
        "recall_fake": recall_score(targets, preds, pos_label=1),
        "f1_fake": f1_score(targets, preds, pos_label=1),
        "train_time_sec": round(train_time, 2),
        "test_latency_sec": round(eval_time, 2),
    }


def save_result_to_csv(result_dict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            # Replace existing row for the model if present, otherwise append
            df_existing = df_existing[df_existing["model"] != result_dict["model"]]
            df_updated = pd.concat([df_existing, pd.DataFrame([result_dict])], ignore_index=True)
        except Exception:
            df_updated = pd.DataFrame([result_dict])
    else:
        df_updated = pd.DataFrame([result_dict])

    df_updated.to_csv(csv_path, index=False)
    print(f"\n[PROGRESS SAVED] Appended/Updated '{result_dict['model']}' in: {csv_path}")


# ==========================================
# Main Execution Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Procedural Multi-Model Trainer")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV")
    parser.add_argument("--models", nargs="+", default=["distilbert", "deberta-v3", "electra", "mobilebert", "bilstm"],
                        help="Models: distilbert deberta-v3 electra mobilebert longformer bilstm distillation all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=None, help="Custom learning rate (e.g. 2e-5, 3e-5, 5e-5). If None, auto-tuned per model.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio over total steps (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 weight decay regularization (default: 0.01)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--output_dir", default="./all_models_output")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, "all_models_results.csv")
    train_df, val_df, test_df = load_and_split_data(args.input, sample=args.sample)

    selected_models = args.models
    if "all" in selected_models:
        selected_models = ["distilbert", "deberta-v3", "electra", "mobilebert", "bilstm", "distillation"]

    results = []

    for m in selected_models:
        res = None
        if m in MODEL_PRESETS:
            res, _ = train_transformer_model(m, MODEL_PRESETS[m], train_df, val_df, test_df, args)
        elif m == "bilstm":
            res = train_bilstm_procedural(train_df, val_df, test_df, args)
        elif m == "distillation":
            teacher_dir = os.path.join(args.output_dir, "deberta-v3")
            if not os.path.exists(teacher_dir):
                teacher_dir = os.path.join(args.output_dir, "distilbert")
            if not os.path.exists(teacher_dir):
                print("Training DeBERTa-v3 first to serve as teacher...")
                res_t, _ = train_transformer_model("deberta-v3", MODEL_PRESETS["deberta-v3"], train_df, val_df, test_df, args)
                if res_t:
                    save_result_to_csv(res_t, summary_csv)
                teacher_dir = os.path.join(args.output_dir, "deberta-v3")

            res = train_distillation_procedural(teacher_dir, train_df, val_df, test_df, args)

        if res:
            results.append(res)
            save_result_to_csv(res, summary_csv)

    if os.path.exists(summary_csv):
        df_res = pd.read_csv(summary_csv)
        print("\n" + "="*80)
        print("  CURRENT MULTI-MODEL COMPARISON SUMMARY")
        print("="*80)
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
