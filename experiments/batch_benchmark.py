"""Benchmark the running API on a labelled multilingual batch.

This is how we measured the live system during the demo. It builds a labelled batch by
translating known NORMAL / ABUSIVE / HATE sentences into the five languages through the
API's own /translate endpoint, sends them to /score, and reports the metrics that matter
for moderation:

  * per-language accuracy,
  * HATE recall (how much hate is caught),
  * ABUSIVE-or-HATE flag rate on abusive input (is toxic content flagged at all),
  * NORMAL false-positive rate (how often clean content is wrongly flagged).

The API must be running and have a translation endpoint configured (see .env.example).

Usage:
    python backend/api.py &                       # or point VOXAFRIK_API at a remote one
    python experiments/batch_benchmark.py
"""
import os
import sys
from collections import defaultdict

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from augment import LANGUAGES, SENTENCES  # noqa: E402

API = os.environ.get("VOXAFRIK_API", "http://localhost:7070")
LABEL_NAMES = {0: "NORMAL", 1: "ABUSIVE", 2: "HATE"}


def translate(text, target):
    try:
        response = requests.post(API + "/translate", json={"text": text, "target": target}, timeout=30)
        return response.json().get("text", "")
    except requests.RequestException:
        return ""


def build_batch():
    items = []
    for label, sentences in SENTENCES.items():
        for english in sentences:
            for language in LANGUAGES:
                translated = translate(english, language)
                if translated:
                    items.append((language, LABEL_NAMES[label], translated))
    print("built", len(items), "labelled messages")
    return items


def main():
    items = build_batch()
    predictions = requests.post(API + "/score", json={"texts": [text for _, _, text in items]}, timeout=180).json()

    per_language = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    hate = [0, 0]
    abusive_flagged = [0, 0]
    normal_false_positive = [0, 0]
    for (language, truth, _), prediction in zip(items, predictions):
        predicted = prediction["category"]
        per_language[language][0] += predicted == truth
        per_language[language][1] += 1
        overall[0] += predicted == truth
        overall[1] += 1
        if truth == "HATE":
            hate[1] += 1
            hate[0] += predicted == "HATE"
        if truth == "ABUSIVE":
            abusive_flagged[1] += 1
            abusive_flagged[0] += predicted != "NORMAL"
        if truth == "NORMAL":
            normal_false_positive[1] += 1
            normal_false_positive[0] += predicted != "NORMAL"

    print("\nlanguage       acc%")
    for language in LANGUAGES:
        correct, total = per_language[language]
        if total:
            print("%-12s  %5.1f  (%d/%d)" % (language, correct / total * 100, correct, total))
    print("-" * 32)
    ratio = lambda pair: pair[0] / pair[1] * 100 if pair[1] else 0.0
    print("overall        %5.1f  (%d/%d)" % (ratio(overall), overall[0], overall[1]))
    print("hate recall    %5.1f  (%d/%d)" % (ratio(hate), hate[0], hate[1]))
    print("abusive flagged%5.1f  (%d/%d)" % (ratio(abusive_flagged), abusive_flagged[0], abusive_flagged[1]))
    print("normal false+  %5.1f  (%d/%d)" % (ratio(normal_false_positive), normal_false_positive[0], normal_false_positive[1]))


if __name__ == "__main__":
    main()
