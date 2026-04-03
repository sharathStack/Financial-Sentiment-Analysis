"""
model.py — DistilBERT fine-tuning setup: tokeniser, tokenisation, model init
"""

import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score
import config


def get_tokenizer():
    return AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)


def tokenize_dataset(train_ds, val_ds, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=config.MAX_LEN)

    train_ds = train_ds.map(_tok, batched=True).rename_column("label", "labels")
    val_ds   = val_ds.map(_tok, batched=True).rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])
    return train_ds, val_ds


def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def get_training_args():
    return TrainingArguments(
        output_dir             = "./fin_sentiment_checkpoints",
        num_train_epochs       = config.EPOCHS,
        per_device_train_batch_size = config.BATCH_SIZE,
        per_device_eval_batch_size  = config.BATCH_SIZE,
        learning_rate          = config.LEARNING_RATE,
        evaluation_strategy    = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "f1_macro",
        logging_dir            = "./logs",
        seed                   = config.SEED,
        report_to              = "none",
    )
