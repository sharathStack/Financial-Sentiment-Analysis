Sentiment Analysis on Financial News (HuggingFace DistilBERT)
![HuggingFace](https://img.shields.io/badge/NLP-HuggingFace-yellow)
![PyTorch](https://img.shields.io/badge/DL-PyTorch-ee4c2c)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
> Fine-tune DistilBERT on financial headlines for 3-class sentiment (Positive / Neutral / Negative). Tracked with MLflow. Deployed via Azure ML endpoint.
Project Structure
```
DS2_FinSentiment__config.py       ← Model + training params
DS2_FinSentiment__data_gen.py     ← 300-headline labelled dataset
DS2_FinSentiment__model.py        ← DistilBERT setup, tokenisation, metrics
DS2_FinSentiment__trainer.py      ← MLflow-tracked training loop
DS2_FinSentiment__inference.py    ← Real-time inference pipeline
DS2_FinSentiment__dashboard.py    ← Results visualisation
DS2_FinSentiment__main.py         ← Entry point
DS2_FinSentiment__requirements.txt
```
Run
```bash
pip install -r DS2_FinSentiment__requirements.txt
python DS2_FinSentiment__main.py
```
Results
+12% accuracy over zero-shot FinBERT baseline
Macro F1 improved from ~0.68 → ~0.81
Model saved to `./fin_sentiment_model` (reload without retraining)
