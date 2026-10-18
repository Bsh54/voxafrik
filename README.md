# VoxAfrik

Real-time hate-speech moderation for African languages, running on IBM LinuxONE.

Team **SAUTILAB** — IBM Z Datathon 2026.

## The problem

Post something hateful in English and a system catches it. Post the same thing in
Hausa, Yoruba or Kinyarwanda and nobody is watching. Platforms moderate the
languages they understand; Africa has over 2,000 of them, and almost none are
covered.

## The solution

VoxAfrik reads a post in an African language and tells a human moderator what it is
looking at — hateful, abusive, or normal — with the words that drove the decision
and an English translation. The model triages and explains; a person always decides.

Supported languages: Swahili, Hausa, Igbo, Kinyarwanda, Yoruba.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py      # AfriHate train/test splits
python backend/augment.py            # optional: broaden clean-input coverage
python backend/api.py                # serves the API and console on :7070
```

Open `http://localhost:7070` and press **Start**.

## Project structure

```
backend/   model, explanation, translation, augmentation, API
frontend/  real-time moderation console
scripts/   environment setup, data download, reverse proxy
docs/      architecture, results, responsible AI
```
