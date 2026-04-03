"""
config.py — Financial News Sentiment Analysis (HuggingFace DistilBERT)
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS       = 3
LABEL2ID         = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL         = {0: "negative", 1: "neutral", 2: "positive"}

# ── Training ───────────────────────────────────────────────────────────────────
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LEARNING_RATE= 2e-5
SEED         = 42
TEST_SIZE    = 0.20

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "financial-sentiment"
MLFLOW_RUN_NAME   = "distilbert-fin-sentiment"

# ── Output ─────────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "./fin_sentiment_model"
CHART_OUTPUT    = "sentiment_results.png"
CHART_DPI       = 150
