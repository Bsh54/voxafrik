# Architecture

VoxAfrik screens social-media posts written in African languages and hands a human
moderator an explainable alert. Everything runs on IBM LinuxONE.

## Pipeline

```
incoming post
   -> classifier (word-embedding + character-CNN ensemble, s390x CPU)
   -> three classes: NORMAL / ABUSIVE / HATE  +  confidence
   -> explanation: English translation + occlusion trigger words
   -> alert routed to a human moderator (never auto-removed)
```

## Components

| Component | File | Role |
|---|---|---|
| Ensemble classifier | `backend/model.py` | Real-time three-class detection |
| Explanation | `backend/explain.py` | Words that drove the decision (occlusion) |
| Translation | `backend/translate.py` | Makes flagged posts legible to any moderator |
| Augmentation | `backend/augment.py` | Broadens coverage to clean, formal input |
| API | `backend/api.py` | Serves the model and the console |
| Console | `frontend/index.html` | Live moderation stream |

## Why IBM Z

Moderation is a volume-and-latency problem: millions of posts a day, each scored
before it spreads. The ensemble scores a post on the s390x CPU in well under a
millisecond, which is the right place for a small model. The heavier generative
layer (IBM Granite) runs on the Telum II on-chip AI accelerator (NNPA) through the
GGML zDNN backend — the right workload for the accelerator. Right workload, right
processor.
