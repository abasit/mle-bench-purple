# Tabular Knowledge Card

## Models

**LightGBM** — fastest, robust defaults, best for high-cardinality categoricals.
Does NOT accept strings — encode with `pd.factorize` on joint train+test first.
```python
cat_cols = [c for c in X.columns if X[c].dtype == 'object']
for c in cat_cols:
    codes, _ = pd.factorize(pd.concat([X[c], X_test[c]]).astype(str), sort=True)
    X[c], X_test[c] = codes[:len(X)].astype('int32'), codes[len(X):].astype('int32')
params = dict(objective='binary', metric='auc', n_estimators=5000, learning_rate=0.02,
              num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
              reg_lambda=0.1, n_jobs=-1, random_state=42, verbose=-1)
```

**CatBoost** — handles string categoricals natively via `cat_features=`. Safest for
mixed-type data. Use only safe params: iterations, learning_rate, depth, l2_leaf_reg,
random_seed, verbose, cat_features, early_stopping_rounds, eval_metric, loss_function.
```python
cat_features = [c for c in X.columns if X[c].dtype == 'object']
model = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                           cat_features=cat_features, early_stopping_rounds=200,
                           random_seed=42, verbose=0)
```

**XGBoost** — strong alternative, requires one-hot or frequency encoding for categoricals.
```python
params = dict(n_estimators=5000, learning_rate=0.02, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
              tree_method='hist', random_state=42, n_jobs=-1)
```

## Cross-validation

| Data type | CV scheme |
|---|---|
| Classification (IID) | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Regression (IID) | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| Grouped rows (same entity) | `GroupKFold(n_splits=5)` |
| Imbalanced + grouped | `StratifiedGroupKFold(n_splits=5)` |
| Time-ordered | `TimeSeriesSplit(n_splits=5)` — NO shuffle |

CV must mirror the train/test relationship. If test was stratified, use stratified CV.

## Feature engineering recipes

**Datetime expansion** (`fe:datetime_expansion`):
```python
dt = pd.to_datetime(df[col], errors='coerce')
df[col+'_year'] = dt.dt.year; df[col+'_month'] = dt.dt.month
df[col+'_dow'] = dt.dt.dayofweek; df[col+'_hour'] = dt.dt.hour
```

**Target encoding OOF** (`fe:target_encoding_oof`):
```python
from sklearn.model_selection import KFold
te_col = np.zeros(len(train))
kf = KFold(5, shuffle=True, random_state=42)
for tr_i, va_i in kf.split(train):
    means = train.iloc[tr_i].groupby(col)[TARGET].mean()
    te_col[va_i] = train[col].iloc[va_i].map(means)
train[col+'_te'] = te_col
test[col+'_te'] = test[col].map(train.groupby(col)[TARGET].mean())
```

**Frequency encoding** (`fe:frequency_encoding`):
```python
freq = pd.concat([train[col], test[col]]).value_counts(normalize=True).to_dict()
train[col+'_freq'] = train[col].map(freq)
test[col+'_freq'] = test[col].map(freq)
```

**Split delimited strings** (`fe:delimited_split`):
For columns like `'A/23/S'`, detect the delimiter and split:
```python
parts = df[col].astype(str).str.split('/', expand=True)
for i, name in enumerate(['part0', 'part1', 'part2']):
    df[col+'_'+name] = parts[i] if i < parts.shape[1] else 'missing'
```

**Interaction features** (`fe:interaction_features`):
```python
top_num = X[numeric_cols].corrwith(y).abs().nlargest(5).index.tolist()
for i, a in enumerate(top_num):
    for b in top_num[i+1:]:
        X[f'{a}_x_{b}'] = X[a] * X[b]
        X[f'{a}_div_{b}'] = X[a] / (X[b] + 1e-8)
```

**Aggregation features** (`fe:aggregation_groupby`):
```python
for cat_col in cat_cols:
    for num_col in num_cols[:3]:
        agg = train.groupby(cat_col)[num_col].agg(['mean','std']).rename(
            columns={'mean': f'{cat_col}_{num_col}_mean', 'std': f'{cat_col}_{num_col}_std'})
        train = train.merge(agg, on=cat_col, how='left')
        test = test.merge(agg, on=cat_col, how='left')
```

## Hyperparameter tuning (`hp:optuna`)

```python
import optuna
def objective(trial):
    p = {
        'learning_rate': trial.suggest_float('lr', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('nl', 15, 127),
        'min_child_samples': trial.suggest_int('mcs', 5, 100),
        'subsample': trial.suggest_float('ss', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('ra', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('rl', 1e-3, 10, log=True),
    }
    # train with early stopping, return val metric
    ...
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, timeout=300)
```

## Ensembling

**Seed averaging** (`ensemble:seed_averaging`): Train same model with 5 seeds, average predictions.
```python
preds = np.zeros(len(test))
for seed in [42, 123, 456, 789, 1024]:
    model = LGBMClassifier(**params, random_state=seed)
    model.fit(X, y); preds += model.predict_proba(X_test)[:,1] / 5
```

**CV fold averaging** (`ensemble:cv_fold_averaging`): Average per-fold test predictions.

**Stacking** (`ensemble:stacking`): Train a meta-model (logistic regression / ridge) on
OOF predictions from base models (LightGBM, CatBoost, XGBoost).

**Rank averaging** (`ensemble:rank_average`): `pd.Series(preds).rank(pct=True)` per model,
then average ranks. Robust to miscalibration across models.

## Post-processing

**Threshold tuning** (`post:threshold_tuning`):
```python
from sklearn.metrics import f1_score
best_t, best_f1 = 0.5, 0
for t in np.arange(0.3, 0.7, 0.01):
    f1 = f1_score(y_val, (oof > t).astype(int))
    if f1 > best_f1: best_t, best_f1 = t, f1
```

## Common pitfalls

- Never use `astype('category')` — use `astype(str).fillna('missing')` instead.
- Never feed string columns to StandardScaler or numeric models without encoding.
- Never fit encoders on test labels. Use `pd.factorize` on joint train+test.
- Never call `.astype(int)` on `'True'`/`'False'` strings — use `.map({'True':1,'False':0})`.
- Drop ID columns and the target from features before fitting.
- Use `predict_proba()` not `predict()` for probability-based metrics (AUC, logloss).
