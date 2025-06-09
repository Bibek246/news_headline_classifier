#!/usr/bin/env python3
# news_headline_classifier.py
# Complete pipeline: AG News classification with TF-IDF+LR, MLP, and optimized DistilBERT,
# including full validation & test metrics and a final comparison chart.

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed
)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# 2. Constants & Device
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)

# 3. Load & subsample AG News
def load_data_hf(subset_size=50000):
    ds = load_dataset("ag_news")
    train = ds["train"].shuffle(seed=SEED).select(range(subset_size))
    test  = ds["test"]
    return train, test

train_ds, test_ds = load_data_hf(subset_size=50000)

# 4. Prepare train/val split
def prepare_splits(ds, val_size=0.1):
    """Split dataset into train/validation using HF ag_news fields."""
    # The HuggingFace ``ag_news`` dataset provides a single ``text`` field and
    # zero-indexed ``label`` values.  The original implementation expected the
    # Kaggle CSV format (``Title``/``Description`` and 1-indexed labels), which
    # results in a ``KeyError`` when using the HF dataset.  Use the correct
    # fields here.
    texts = list(ds["text"])
    labels = list(ds["label"])
    return train_test_split(texts, labels,
                            test_size=val_size,
                            random_state=SEED,
                            stratify=labels)

X_train, X_val, y_train, y_val = prepare_splits(train_ds)

# 5. Baseline: TF-IDF + Logistic Regression
def train_baseline(X_train, y_train, X_val, y_val):
    tfidf = TfidfVectorizer(lowercase=True,
                            stop_words="english",
                            max_features=20000,
                            ngram_range=(1,2))
    Xtr = tfidf.fit_transform(X_train)
    Xvl = tfidf.transform(X_val)

    lr = LogisticRegression(max_iter=1000,
                            class_weight="balanced",
                            random_state=SEED)
    lr.fit(Xtr, y_train)

    preds = lr.predict(Xvl)
    print(f"[Baseline] Val Acc: {accuracy_score(y_val, preds):.4f}")
    print(classification_report(y_val, preds, target_names=CATEGORIES))

    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CATEGORIES,
                yticklabels=CATEGORIES,
                cmap="Blues")
    plt.title("Baseline Confusion Matrix (Val)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()

    import joblib
    joblib.dump(tfidf, "models/tfidf.pkl")
    joblib.dump(lr,    "models/logreg.pkl")
    return tfidf, lr

tfidf, lr_model = train_baseline(X_train, y_train, X_val, y_val)

# 6. MLP on TF-IDF
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(CATEGORIES))
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(tfidf, X_train, y_train, X_val, y_val, epochs=3):
    Xtr = torch.from_numpy(tfidf.transform(X_train).toarray()).float()
    ytr = torch.tensor(y_train).long()
    Xvl = torch.from_numpy(tfidf.transform(X_val).toarray()).float()
    yvl = torch.tensor(y_val).long()

    train_ds = TensorDataset(Xtr, ytr)
    val_ds   = TensorDataset(Xvl, yvl)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    model = MLPClassifier(input_dim=Xtr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
        acc = correct / len(val_loader.dataset)
        print(f"[MLP] Epoch {epoch} — Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
        best_acc = max(best_acc, acc)
    return model, best_acc

mlp_model, mlp_val_acc = train_mlp(tfidf, X_train, y_train, X_val, y_val)

# 7. DistilBERT fine‑tuning
def fine_tune_distilbert(X_train, y_train, X_val, y_val):
    hf_train = Dataset.from_dict({"text": X_train, "label": y_train})
    hf_val   = Dataset.from_dict({"text": X_val,   "label": y_val})
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["text"],
                         padding="max_length",
                         truncation=True,
                         max_length=32)
    hf_train = hf_train.map(tokenize_fn, batched=True)
    hf_val   = hf_val.map(tokenize_fn, batched=True)
    hf_train.set_format("torch", columns=["input_ids","attention_mask","label"])
    hf_val.set_format("torch",   columns=["input_ids","attention_mask","label"])

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(CATEGORIES)
    ).to(DEVICE)

    args = TrainingArguments(
        output_dir="models/distilbert",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available()
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        tokenizer=tokenizer
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"[DistilBERT] Val loss: {eval_metrics['eval_loss']:.4f}")

    preds_out = trainer.predict(hf_val)
    y_pred = np.argmax(preds_out.predictions, axis=1)
    y_true = preds_out.label_ids

    print("=== DistilBERT Validation Metrics ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=CATEGORIES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CATEGORIES,
                yticklabels=CATEGORIES,
                cmap="Blues")
    plt.title("DistilBERT Confusion Matrix (Val)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()

    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    return trainer, hf_val

trainer, hf_val = fine_tune_distilbert(X_train, y_train, X_val, y_val)

# 8. Evaluate baseline on TEST
# The test split from the HF dataset uses the same ``text`` and ``label``
titles_test = test_ds["text"]
X_test = list(titles_test)
y_test = list(test_ds["label"])

Xte = tfidf.transform(X_test)
preds_base = lr_model.predict(Xte)
print(f"[Baseline] Test Acc: {accuracy_score(y_test, preds_base):.4f}")
print(classification_report(y_test, preds_base, target_names=CATEGORIES))

# 9. Evaluate DistilBERT on TEST
tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
model     = DistilBertForSequenceClassification.from_pretrained("models/distilbert").to(DEVICE)

hf_test = Dataset.from_dict({"text": X_test, "label": y_test})
hf_test = hf_test.map(lambda b: tokenizer(b["text"], padding="max_length", truncation=True, max_length=32), batched=True)
hf_test.set_format("torch", columns=["input_ids","attention_mask","label"])

test_trainer = Trainer(model=model, args=TrainingArguments(output_dir="models/distilbert", per_device_eval_batch_size=64))
preds_test = test_trainer.predict(hf_test)
y_pred_test = np.argmax(preds_test.predictions, axis=1)
y_true_test = preds_test.label_ids

print("=== DistilBERT Test Metrics ===")
print("Accuracy:", accuracy_score(y_true_test, y_pred_test))
print(classification_report(y_true_test, y_pred_test, target_names=CATEGORIES))

cm_test = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(6,5))
sns.heatmap(cm_test, annot=True, fmt="d",
            xticklabels=CATEGORIES,
            yticklabels=CATEGORIES,
            cmap="Blues")
plt.title("DistilBERT Confusion Matrix (Test)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.show()

# 10. Final model comparison
results = {
    "LR Baseline (Test)": accuracy_score(y_test, preds_base),
    "MLP (Val)": mlp_val_acc,
    "DistilBERT (Test)": accuracy_score(y_true_test, y_pred_test)
}
plt.figure(figsize=(6,4))
plt.bar(results.keys(), results.values())
plt.ylim(0.8,1.0)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.show()
