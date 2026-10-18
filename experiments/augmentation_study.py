"""Measure what synthetic augmentation does for clean, formally phrased input.

The classifier is trained on real, noisy social-media text. A moderator demo, however,
often types a clean sentence (sometimes written in English and translated), and early
on the model missed hate in that clean register even though it did well on the real
test set. This study quantifies the gap and the fix.

It trains the ensemble twice on the real AfriHate training data:
  * BASELINE: real data only.
  * AUGMENTED: real data plus generated-and-translated examples.
Both are evaluated on two sets:
  * the real AfriHate test split (held out, noisy, in-domain), and
  * a FRESH set of clean generated sentences translated into the five languages,
    kept completely out of training, standing in for live demo input.

This is the experiment behind the augmentation table in docs/results.md.

Usage:
    python experiments/augmentation_study.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from augment import LANGUAGES, SENTENCES  # noqa: E402  reuse the single sentence source
from data import load_split  # noqa: E402
from model import Ensemble  # noqa: E402
from translate import translate  # noqa: E402

DATA_DIR = os.environ.get("VOXAFRIK_DATA", os.path.join(ROOT, "data", "afrihate"))
TRAIN_PER_CLASS = 14  # first N sentences per class feed augmentation, the rest stay fresh


def translate_split():
    """Translate the seed sentences, splitting each class into train and fresh held-out."""
    train_rows, fresh_rows = [], []
    for label, sentences in SENTENCES.items():
        for index, english in enumerate(sentences):
            bucket = train_rows if index < TRAIN_PER_CLASS else fresh_rows
            for language in LANGUAGES:
                try:
                    translated = translate(english, language)
                except Exception:
                    translated = ""
                if translated.strip():
                    bucket.append((translated.strip(), label))
    print("augmentation rows: %d, fresh held-out rows: %d" % (len(train_rows), len(fresh_rows)))
    return train_rows, fresh_rows


def hate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)[2] * 100


def evaluate(model, test, fresh_x, fresh_y, name):
    real_pred = model.predict_proba(test["text"].astype(str).values).argmax(1)
    real_y = test["label"].astype(int).values
    fresh_pred = model.predict_proba(fresh_x).argmax(1)
    print("%-10s | real test acc %.1f hate %.1f | fresh acc %.1f hate %.1f" % (
        name,
        accuracy_score(real_y, real_pred) * 100, hate_recall(real_y, real_pred),
        accuracy_score(fresh_y, fresh_pred) * 100, hate_recall(fresh_y, fresh_pred)))


def main():
    train = load_split(DATA_DIR, "train")
    test = load_split(DATA_DIR, "test")
    augment_rows, fresh_rows = translate_split()
    fresh_x = np.array([text for text, _ in fresh_rows])
    fresh_y = np.array([label for _, label in fresh_rows])

    real_x = train["text"].astype(str).tolist()
    real_y = train["label"].astype(int).tolist()
    evaluate(Ensemble().fit(real_x, real_y), test, fresh_x, fresh_y, "BASELINE")

    augmented_x = real_x + [text for text, _ in augment_rows]
    augmented_y = real_y + [label for _, label in augment_rows]
    evaluate(Ensemble().fit(augmented_x, augmented_y), test, fresh_x, fresh_y, "AUGMENTED")


if __name__ == "__main__":
    main()
