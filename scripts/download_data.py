"""Download the AfriHate train and test splits for the five supported languages."""
import os
from datasets import load_dataset

LANGUAGES = ["swa", "hau", "ibo", "kin", "yor"]
OUTPUT_DIR = "data/afrihate"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for code in LANGUAGES:
        for split in ["train", "test"]:
            dataset = load_dataset("mteb/AfriHateClassification", code, split=split)
            frame = dataset.to_pandas()[["text", "label"]]
            frame.to_csv(f"{OUTPUT_DIR}/{code}_{split}.csv", index=False)
            print(code, split, len(frame))


if __name__ == "__main__":
    main()
