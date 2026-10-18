# Measured results

All numbers come from the official AfriHate test splits, never seen during training.
Reproduce with `python backend/evaluate.py data/afrihate`.

## Ensemble on the AfriHate test set

| Language | Accuracy | Macro-F1 | Hate recall |
|---|---|---|---|
| Swahili | 88.1% | 0.869 | 72.4% |
| Igbo | 81.9% | 0.796 | 69.9% |
| Kinyarwanda | 72.3% | 0.688 | 58.6% |
| Hausa | 73.1% | 0.578 | 15.5% |
| Yoruba | 76.6% | 0.567 | 3.8% |
| **Overall** | **81.3%** | **0.787** | **55.4%** |

We report precision/recall per class and per language, never a single global
accuracy: the classes are imbalanced and a "never flag" model would score high
accuracy while catching no hate.

## Augmentation

The classifier is trained on real, noisy social-media text. On clean, formally
phrased input (for example a message typed in English and translated), hate recall
dropped to 40%. Adding generated-and-translated examples to training raised it to
77% on held-out fresh input, without hurting the real test set.

| | Real test (hate recall) | Fresh input (hate recall) |
|---|---|---|
| Baseline | 74.2% | 40.0% |
| Augmented | 72.9% | 76.7% |

## Latency

The ensemble scores on the order of 0.04 ms per post on the s390x CPU
(about 25,000 posts/second), which is what makes continuous real-time screening
possible.

## Honest limits

- Yoruba hate recall is low because the training split contains very few hate
  examples; this is a data-scarcity limit, reported rather than hidden.
- AfriHate's target-attribute labels (ethnicity, religion, gender, ...) are
  described in the paper but are not part of the released dataset, so a target
  classifier is left as future work.
