# NLP Knowledge Card

## Models

| Model | HuggingFace name | Notes |
|---|---|---|
| DeBERTa-v3-base | `microsoft/deberta-v3-base` | Strongest encoder for classification |
| RoBERTa-base | `roberta-base` | Solid alternative |
| XLM-RoBERTa | `xlm-roberta-base` | Multilingual |

No GPU? Use **TF-IDF + LightGBM** — often within 2-3% of BERT on small datasets:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
X_tr = vec.fit_transform(train_text)
model = LGBMClassifier(n_estimators=2000, learning_rate=0.05)
```

## Training recipe (HuggingFace)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base', num_labels=NUM_LABELS)

args = TrainingArguments(
    output_dir='./tmp', num_train_epochs=3,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy='epoch', save_strategy='no',
    fp16=True, report_to='none', seed=42)
```
- max_length: start with 128 — many tasks fit there and train 4x faster.
- For long documents (>512 tokens): split into windows, predict each, mean-pool.

## Tricks

**Pseudo-labeling**: Train ensemble → predict test → keep confident (>0.95) → retrain.

**Layer-wise LR decay**: Lower layers get smaller LR (multiply by 0.9 per layer).

**Multi-sample dropout**: Apply dropout N times on the CLS embedding, average the logits.

## Cross-validation

StratifiedKFold(5) by label. For multi-label use MultilabelStratifiedKFold.

## Pitfalls

- Never set max_length=512 blindly — profile your data first.
- Never use `fp16=True` without a GPU that supports it.
- Never forget to mask padding tokens for mean pooling.
- Never skip `seed_everything()` — HuggingFace Trainer uses its own seed.
