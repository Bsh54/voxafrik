"""Flask API that serves the VoxAfrik model and the moderation console on IBM LinuxONE."""
import os
import random
import time
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from data import load_split, LANGUAGES, LABELS
from model import Ensemble

DATA_DIR = os.environ.get("VOXAFRIK_DATA", "data/afrihate")
AUGMENT_PATH = os.environ.get("VOXAFRIK_AUGMENT", "augment.csv")
FRONTEND_DIR = os.environ.get("VOXAFRIK_FRONTEND", os.path.join(os.path.dirname(__file__), "..", "frontend"))

train = load_split(DATA_DIR, "train")
test = load_split(DATA_DIR, "test")
texts = train["text"].astype(str).tolist()
labels = train["label"].astype(int).tolist()
if os.path.exists(AUGMENT_PATH):
    extra = pd.read_csv(AUGMENT_PATH).dropna(subset=["text", "label"])
    texts += extra["text"].astype(str).tolist()
    labels += extra["label"].astype(int).tolist()
model = Ensemble().fit(texts, labels)

app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ready", "languages": list(LANGUAGES.values())})


@app.route("/sample")
def sample():
    count = min(int(request.args.get("n", 30)), 300)
    rows = test.sample(count).reset_index(drop=True)
    start = time.perf_counter()
    probs = model.predict_proba(rows["text"].astype(str).values)
    per_message = (time.perf_counter() - start) * 1000 / max(len(rows), 1)
    results = []
    for index_, row in rows.iterrows():
        proba = probs[index_]
        label = int(proba.argmax())
        results.append({
            "id": "ds-" + str(random.randint(0, 10 ** 9)),
            "dialect": LANGUAGES[row["lang"]],
            "origin": "AfriHate dataset",
            "text": str(row["text"]),
            "translation": "",
            "category": LABELS[label],
            "confidence": round(float(proba[label]), 2),
            "isToxic": label != 0,
            "latency": f"{per_message:.3f} ms",
            "tokens": [],
        })
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7070)), threaded=True)
