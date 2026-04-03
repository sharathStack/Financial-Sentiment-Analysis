"""
data_gen.py — Financial headline dataset generator

300-sample labelled dataset of financial news headlines.
Each label has 10 seed headlines × 3 augmentations = 90 per class.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import config

HEADLINES = {
    "positive": [
        "Company reports record-breaking quarterly profits",
        "Stock surges after strong earnings beat expectations",
        "Firm announces major expansion into Asian markets",
        "Revenue growth exceeds analyst forecasts by 20 percent",
        "New product launch drives investor confidence higher",
        "Merger deal approved boosting shareholder value significantly",
        "Central bank signals rate cuts fueling broad market rally",
        "Tech giant unveils AI platform sending shares soaring",
        "Unemployment falls to historic low boosting consumer spending",
        "Dividend increased for the fifth consecutive year",
    ],
    "neutral": [
        "Federal Reserve keeps interest rates unchanged at current levels",
        "Company maintains full-year guidance amid economic uncertainty",
        "Markets open flat ahead of key economic data release",
        "CFO steps down in planned leadership transition",
        "Quarterly results in line with analyst consensus estimates",
        "Firm completes routine debt refinancing process",
        "Board of directors approves annual operating budget plan",
        "Trade talks between US and China continue as scheduled",
        "Oil prices remain steady after overnight trading session",
        "Earnings release scheduled for end of the current month",
    ],
    "negative": [
        "Profit warning issued as sales miss targets significantly",
        "Stock plunges after deeply disappointing earnings report",
        "Layoffs announced as company restructures operations globally",
        "Regulatory probe launched into accounting practices",
        "Supply chain disruptions weigh heavily on quarterly results",
        "Revenue declines for third consecutive quarter raising alarm",
        "Credit rating downgraded amid rising debt burden concerns",
        "Executive arrested on insider trading charges",
        "Market selloff deepens on growing recession fears",
        "Bank reports surge in non-performing loan portfolio",
    ],
}


def generate():
    rows = []
    np.random.seed(config.SEED)
    for label, headlines in HEADLINES.items():
        lid = config.LABEL2ID[label]
        for h in headlines:
            rows.append({"text": h,            "label": lid})
            rows.append({"text": h.lower(),    "label": lid})
            rows.append({"text": h + ".",      "label": lid})

    df = pd.DataFrame(rows).sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=config.TEST_SIZE,
                                         random_state=config.SEED, stratify=df["label"])
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
    return train_ds, val_ds, df
