"""
trainer.py — MLflow-tracked training loop for DistilBERT fine-tuning
"""

import json
import mlflow
from transformers import Trainer
import config
from model import (get_tokenizer, tokenize_dataset, build_model,
                   compute_metrics, get_training_args)


def run_training(train_ds, val_ds):
    """Fine-tune DistilBERT with MLflow experiment tracking."""
    tokenizer   = get_tokenizer()
    train_tok, val_tok = tokenize_dataset(train_ds, val_ds, tokenizer)
    model       = build_model()
    args        = get_training_args()

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_tok,
        eval_dataset    = val_tok,
        tokenizer       = tokenizer,
        compute_metrics = compute_metrics,
    )

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=config.MLFLOW_RUN_NAME):
        mlflow.log_params({
            "model":         config.MODEL_CHECKPOINT,
            "epochs":        config.EPOCHS,
            "batch_size":    config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "max_len":       config.MAX_LEN,
        })

        print("\n── Training ──────────────────────────────────────")
        trainer.train()
        eval_results = trainer.evaluate()
        mlflow.log_metrics(eval_results)

        print("\n── Eval Results ──────────────────────────────────")
        print(json.dumps(eval_results, indent=2))

    return model, tokenizer, eval_results
