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

## Results

On the official AfriHate test splits the ensemble reaches **81.3% accuracy** and
**0.787 macro-F1**. Full per-language numbers and the augmentation study are in
[docs/results.md](docs/results.md).

## Running on IBM Z

The real-time classifier runs on the s390x CPU; the generative reasoning layer
(IBM Granite) runs on the Telum II on-chip AI accelerator. See
[docs/architecture.md](docs/architecture.md).

## Responsible AI

Three classes instead of a binary, per-language metrics, and a human in the loop on
every decision. See [docs/responsible_ai.md](docs/responsible_ai.md).

## License

MIT — see [LICENSE](LICENSE). Dataset: AfriHate (Muhammad et al., NAACL 2025).
