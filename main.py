"""
main.py — Financial Sentiment Analysis entry point
"""

import config
from data_gen  import generate
from trainer   import run_training
from inference import build_pipeline, predict_batch, DEMO_HEADLINES
from dashboard import plot_results


def main():
    print("=" * 55)
    print("  FINANCIAL NEWS SENTIMENT ANALYSIS")
    print("  DistilBERT Fine-Tuning + MLflow Tracking")
    print("=" * 55)

    # 1. Data
    print("\n[1] Generating dataset...")
    train_ds, val_ds, df = generate()

    # 2. Train
    print("\n[2] Fine-tuning DistilBERT...")
    model, tokenizer, eval_results = run_training(train_ds, val_ds)

    # 3. Save model
    print(f"\n[3] Saving model → {config.MODEL_SAVE_PATH}")
    model.save_pretrained(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)

    # 4. Inference demo
    print("\n[4] Running inference on demo headlines...")
    clf         = build_pipeline(model, tokenizer)
    predictions = predict_batch(clf, DEMO_HEADLINES)

    print("\n── Inference Demo ──────────────────────────────────")
    for p in predictions:
        print(f"  [{p['sentiment']:8s}] ({p['confidence']:.3f})  {p['headline']}")

    # 5. Dashboard
    print("\n[5] Generating dashboard...")
    plot_results(eval_results, predictions)

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
