"""Measure a large multilingual transformer (AfroXLMR) as an alternative classifier.

Before settling on the small ensemble, we checked what a big pretrained multilingual
transformer could do on the same task. This script encodes every AfriHate message into
a sentence embedding with AfroXLMR, then trains a simple logistic-regression head on
top, both per language and pooled across all five languages. It reports accuracy,
macro-F1 and hate recall so the numbers are directly comparable to backend/evaluate.py.

What we learned: AfroXLMR embeddings plus a linear head are competitive on accuracy but
much heavier at inference, which does not fit a model that has to score a live stream in
real time on the CPU. The small ensemble gives almost the same quality at a tiny
fraction of the latency, so it is what the live system uses. This file keeps the
comparison reproducible.

Usage:
    pip install -r experiments/requirements.txt
    python experiments/afroxlmr_probe.py                 # Davlan/afro-xlmr-base
    python experiments/afroxlmr_probe.py Davlan/afro-xlmr-mini
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("VOXAFRIK_DATA", os.path.join(ROOT, "data", "afrihate"))
LANGUAGES = ["swa", "hau", "ibo", "kin", "yor"]
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Davlan/afro-xlmr-base"


def mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(1) / mask.sum(1)


def build_encoder():
    torch.set_num_threads(3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()

    @torch.no_grad()
    def encode(texts, batch_size=32):
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=96, return_tensors="pt")
            vectors.append(mean_pool(model(**tokens).last_hidden_state, tokens["attention_mask"]).numpy())
        return np.vstack(vectors)

    return encode


def load(code, split):
    frame = pd.read_csv(os.path.join(DATA_DIR, f"{code}_{split}.csv")).dropna(subset=["text", "label"])
    return frame["text"].astype(str).tolist(), frame["label"].astype(int).values


def scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    hate_recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)[2] * 100
    return accuracy, macro_f1, hate_recall


def main():
    print("encoding with", MODEL_NAME)
    encode = build_encoder()
    train = {code: (encode(load(code, "train")[0]), load(code, "train")[1]) for code in LANGUAGES}
    test = {code: (encode(load(code, "test")[0]), load(code, "test")[1]) for code in LANGUAGES}

    print("\nper-language head (AfroXLMR + balanced logistic regression)")
    print("lang    acc%   macroF1  hate%")
    for code in LANGUAGES:
        head = LogisticRegression(max_iter=3000, class_weight="balanced").fit(*train[code])
        accuracy, macro_f1, hate_recall = scores(test[code][1], head.predict(test[code][0]))
        print("%-6s  %5.1f  %7.3f  %5.1f" % (code, accuracy, macro_f1, hate_recall))

    print("\npooled multilingual head")
    pooled_x = np.vstack([train[code][0] for code in LANGUAGES])
    pooled_y = np.concatenate([train[code][1] for code in LANGUAGES])
    head = LogisticRegression(max_iter=3000, class_weight="balanced").fit(pooled_x, pooled_y)
    all_true, all_pred = [], []
    print("lang    acc%   macroF1  hate%")
    for code in LANGUAGES:
        predictions = head.predict(test[code][0])
        all_true.append(test[code][1])
        all_pred.append(predictions)
        accuracy, macro_f1, hate_recall = scores(test[code][1], predictions)
        print("%-6s  %5.1f  %7.3f  %5.1f" % (code, accuracy, macro_f1, hate_recall))
    accuracy, macro_f1, hate_recall = scores(np.concatenate(all_true), np.concatenate(all_pred))
    print("%-6s  %5.1f  %7.3f  %5.1f" % ("ALL", accuracy, macro_f1, hate_recall))


if __name__ == "__main__":
    main()
