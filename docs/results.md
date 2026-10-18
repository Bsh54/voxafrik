# Measured results

All numbers come from the official AfriHate test splits, held out and never seen during
training. Reproduce the headline table with:

```bash
python backend/evaluate.py
```

## Ensemble on the AfriHate test set

This is a real run of `backend/evaluate.py` on the held-out test split. Training uses a
fixed seed, so the run is deterministic and these numbers come back the same each time.

| Language | Accuracy | Macro-F1 | Hate recall |
|---|---|---|---|
| Swahili | 85.6% | 0.839 | 87.4% |
| Igbo | 84.6% | 0.822 | 82.8% |
| Hausa | 73.2% | 0.685 | 49.5% |
| Yoruba | 76.8% | 0.579 | 5.7% |
| Kinyarwanda | 67.9% | 0.650 | 90.9% |
| **Overall** | **80.0%** | **0.782** | **74.2%** |

We report precision and recall per class and per language, never a single global
accuracy. The classes are imbalanced, and a model that simply never flags anything would
post a high accuracy while catching no hate at all. The per-language view is the honest
one, and it shows plainly where the system is strong (Swahili, Igbo) and where it is weak
(Yoruba in particular, where hate examples in training are very few).

## Why the ensemble and not a large transformer

Before choosing the ensemble we measured a large pretrained multilingual transformer,
AfroXLMR, with a logistic-regression head on its embeddings, both per language and pooled.
It is competitive on quality but far heavier at inference, which does not fit a model that
must score a live stream in real time on the CPU. The small ensemble reaches comparable
quality at a tiny fraction of the latency. The comparison is reproducible in
[../experiments/afroxlmr_probe.py](../experiments/afroxlmr_probe.py).

## Augmentation for clean input

The classifier is trained on real, noisy social-media text. On clean, formally phrased
input (for example a message typed in English and then translated), hate recall dropped to
about 40 percent even though the real test score was fine. Adding a small set of
generated-and-translated examples to training raised fresh-input hate recall to about 77
percent without hurting the real test set.

| | Real test (hate recall) | Fresh input (hate recall) |
|---|---|---|
| Baseline | 74.2% | 40.0% |
| Augmented | 72.9% | 76.7% |

The study is reproducible in
[../experiments/augmentation_study.py](../experiments/augmentation_study.py).

## Latency

Measured on the LinuxONE JupyterLab container, the full ensemble (both models) scores a
batch at about **0.22 ms per message, roughly 4,500 messages per second**. This was taken
over several passes on 8,000 messages, keeping the best pass. The container is shared, so a
dedicated LPAR would do better, but even this figure is what makes continuous real-time
screening of a live stream practical. [../ibmz/check_accelerator.py](../ibmz/check_accelerator.py)
reports the throughput on whatever machine it runs on.

## Honest limits

- Yoruba hate recall is low because the training split contains very few hate examples.
  This is a data-scarcity limit, reported rather than hidden.
- Hausa hate recall is also low for the same reason, and it is the next thing more data
  would fix.
- AfriHate's target-attribute labels (ethnicity, religion, gender and so on) are described
  in the paper but are not part of the released dataset, so a target classifier is left as
  future work rather than claimed.
