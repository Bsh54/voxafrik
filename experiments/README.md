# Experiments

These scripts are the research behind VoxAfrik's design choices and the numbers in
[../docs/results.md](../docs/results.md). They are not part of the live system, but they
are what justified it. Each one is self-contained and reproducible.

```bash
pip install -r experiments/requirements.txt
```

## `afroxlmr_probe.py`

Encodes AfriHate with AfroXLMR, a large pretrained multilingual transformer, and trains
a logistic-regression head on the embeddings, per language and pooled. This is the
"should we just use a big transformer?" question. The answer: the big model is
competitive on quality but far too heavy to score a live stream in real time on the CPU,
while the small ensemble reaches almost the same quality at a fraction of the latency.
That is why the live classifier is the ensemble, not AfroXLMR.

```bash
python experiments/afroxlmr_probe.py                  # afro-xlmr-base
python experiments/afroxlmr_probe.py Davlan/afro-xlmr-mini
```

## `augmentation_study.py`

Trains the ensemble with and without generated-and-translated examples and evaluates
both on the real AfriHate test split and on a fresh, clean, held-out set. This produces
the augmentation table in the results doc, where hate recall on clean input rises from
about 40 percent to about 77 percent without hurting the real test set. Needs a
translation endpoint configured (see ../.env.example).

```bash
python experiments/augmentation_study.py
```

## `batch_benchmark.py`

Drives the running API with a labelled multilingual batch and reports per-language
accuracy, hate recall, abusive-flag rate and normal false-positive rate. This is how we
measured the live demo rather than only the offline split.

```bash
python backend/api.py &
python experiments/batch_benchmark.py
```
