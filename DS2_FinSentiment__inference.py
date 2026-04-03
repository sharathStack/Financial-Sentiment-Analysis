"""
inference.py — Real-time inference pipeline + batch prediction
"""

from transformers import pipeline
import config


def build_pipeline(model, tokenizer, device: int = -1):
    """Build a HuggingFace text-classification pipeline."""
    return pipeline("text-classification", model=model,
                    tokenizer=tokenizer, device=device)


def predict_headline(clf, headline: str) -> dict:
    """Predict sentiment for a single headline."""
    result = clf(headline)[0]
    # Map LABEL_0 / LABEL_1 / LABEL_2 → human-readable
    raw_label = result["label"]
    if raw_label.startswith("LABEL_"):
        idx   = int(raw_label.split("_")[1])
        label = config.ID2LABEL[idx]
    else:
        label = raw_label
    return {"headline": headline, "sentiment": label.upper(),
            "confidence": round(result["score"], 4)}


def predict_batch(clf, headlines: list) -> list:
    return [predict_headline(clf, h) for h in headlines]


# ── Demo headlines ─────────────────────────────────────────────────────────────
DEMO_HEADLINES = [
    "Company posts record profit, stock hits all-time high",
    "Layoffs expected as firm struggles with rising costs",
    "Markets remain unchanged ahead of Fed announcement",
    "Earnings miss drives shares down 12 percent in after-hours trading",
    "Firm secures billion-dollar government contract boosting outlook",
]
