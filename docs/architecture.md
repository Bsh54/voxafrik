# Architecture

VoxAfrik screens social-media posts written in African languages and hands a human
moderator an explainable alert. Everything runs on IBM LinuxONE.

## Pipeline

```
incoming post
   -> classifier (word-embedding + character-CNN ensemble, s390x CPU)
   -> three classes: NORMAL / ABUSIVE / HATE, with a confidence score
   -> explanation: English translation plus occlusion trigger words
   -> alert routed to a human moderator (never auto-removed)
```

A post enters at the bottom of the console, crosses the classifier, and either flows
through as clean or slides into the flagged queue, where a moderator can open it, read the
translation and the trigger words, and decide.

## Two layers, two parts of the chip

Moderation is a volume-and-latency problem: millions of posts a day, each one scored
before it spreads. VoxAfrik splits the work so that each layer runs where it belongs.

**Classifier, on the s390x CPU.** The ensemble is intentionally small. A word-embedding
model learns which words and combinations carry abuse or hate; a character-level CNN
reads the raw characters and stays robust to the spelling variation, code-switching and
slang of real social text. Averaging the two is more stable than either alone. Being
small, it scores a post in well under a millisecond on a general-purpose core, which is
exactly what continuous real-time screening needs.

**Generative layer, on the Telum II accelerator.** When a post is flagged, an IBM Granite
model writes a short reason in plain language for the moderator. Generation is heavy
matrix multiplication, the workload the Telum II on-chip AI accelerator (NNPA) is built
for, so it runs there through the GGML zDNN backend rather than on the CPU. Right workload,
right processor. Details and an honest note on model size are in
[../ibmz/README.md](../ibmz/README.md).

## Components

| Component | File | Role |
|---|---|---|
| Ensemble classifier | `backend/model.py` | Real-time three-class detection |
| Explanation | `backend/explain.py` | Words that drove the decision, by occlusion |
| Translation | `backend/translate.py` | Makes flagged posts legible to any moderator |
| Augmentation | `backend/augment.py` | Broadens coverage to clean, formal input |
| Data loading | `backend/data.py` | Reads the AfriHate splits and the label map |
| Evaluation | `backend/evaluate.py` | Per-language accuracy, macro-F1 and hate recall |
| API | `backend/api.py` | Trains at start-up and serves the model and console |
| Console | `frontend/index.html` | Live moderation stream |
| Accelerator | `ibmz/` | Accelerator check and Granite on Telum |

## How the explanation works

The explanation is occlusion-based and needs no extra model. For a flagged post, the
classifier scores the full text, then scores it again with each word removed in turn. The
words whose removal drops the confidence in the predicted class the most are the words
that drove the decision. They are returned with a weight and shown to the moderator. This
is simple, faithful to the model actually in use, and cheap enough to run on every alert.

## Serving and deployment

`backend/api.py` trains the ensemble once at start-up on the AfriHate training data (plus
the augmentation file when present) and then serves four endpoints: `/sample` streams
scored messages to the console, `/score` classifies arbitrary text, `/explain` returns the
translation and trigger words, and `/translate` proxies a translation service. The single
page in `frontend/index.html` is served at `/`.

For a public demo the API sits behind a reverse proxy; `scripts/deploy_proxy.sh` sets up
nginx in front of it, and any HTTPS tunnel can expose that proxy. The proxy is
configuration only and carries no secrets.
