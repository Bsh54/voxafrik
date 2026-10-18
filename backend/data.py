"""Loading and labelling helpers for the AfriHate datasets used by VoxAfrik."""
import pandas as pd

LANGUAGES = {"swa": "Swahili", "hau": "Hausa", "ibo": "Igbo", "kin": "Kinyarwanda", "yor": "Yoruba"}
LABELS = {0: "NORMAL", 1: "ABUSIVE", 2: "HATE"}


def load_split(base_dir, split):
    frames = []
    for code in LANGUAGES:
        frame = pd.read_csv(f"{base_dir}/{code}_{split}.csv").dropna(subset=["text", "label"])
        frames.append(frame.assign(lang=code))
    return pd.concat(frames, ignore_index=True)
