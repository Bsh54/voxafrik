"""Evaluate the ensemble per language on the official AfriHate test splits."""
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, recall_score
from data import load_split, LANGUAGES
from model import Ensemble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "afrihate")


def main(base_dir):
    train = load_split(base_dir, "train")
    test = load_split(base_dir, "test")
    model = Ensemble().fit(train["text"].astype(str).values, train["label"].astype(int).values)
    print("language      acc%   macroF1  hate_recall%")
    for code, name in LANGUAGES.items():
        subset = test[test["lang"] == code]
        labels = subset["label"].astype(int).values
        predictions = model.predict_proba(subset["text"].astype(str).values).argmax(1)
        accuracy = accuracy_score(labels, predictions) * 100
        macro_f1 = f1_score(labels, predictions, average="macro")
        hate_recall = recall_score(labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)[2] * 100
        print("%-12s %5.1f  %7.3f  %6.1f" % (name, accuracy, macro_f1, hate_recall))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VOXAFRIK_DATA", DEFAULT_DATA_DIR))
