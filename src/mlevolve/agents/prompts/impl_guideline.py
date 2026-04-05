"""Implementation guideline."""

import time

import humanize


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    return get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
    )


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
) -> dict:
    """Build implementation guideline from time and config."""
    impl_guideline = [
        f"**Resource Budget**: Time left ≈ {_format_time(tot_time_remaining)} | Steps left = {steps_remaining} | Max execution time per run = {humanize.naturaldelta(exec_timeout)}",
        "",
        "**Note:** Code execution MUST complete within 9 hours (hard limit) — any solution exceeding this will be invalid. Within this constraint, prioritize performance and optimization.",
        "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "",
        "**1. Model Inference for ALL Predictions**",
        "• EVERY prediction (validation & test) MUST come from trained model's forward pass",
        "• Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions",
        "• ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "• ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "• Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        "**2. Generate submission.csv**",
        "• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
        "• Content: Model predictions on ALL test samples",
        "• Format: Follow task description exactly",
        "",
        "**3. Print Validation Metric**",
        "• MUST print: `print(f'Final Validation Score: {score}')`",
        "• Score MUST be computed on hold-out validation set using proper metric formula",
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`",
        "",
        f"📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development."
        + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else ""),
        "",
        "🔴 **CRITICAL API Rules — violations cause immediate crash**:",
        "• LightGBM: `LGBMClassifier(verbosity=-1)` in constructor. `fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])`. NEVER pass `early_stopping_rounds` or `verbose` to fit().",
        "• XGBoost with early stopping: `XGBClassifier(verbosity=0)` in constructor. Pass `early_stopping_rounds=10` and `eval_set=[(X_val, y_val)]` to fit(). When retraining on full data (no val set), do NOT pass early_stopping_rounds at all — omit it or set `best_iteration` manually.",
        "• CatBoost: `CatBoostClassifier(early_stopping_rounds=50, verbose=0)` in constructor. Pass `eval_set=(X_val, y_val)` to fit(). NEVER pass `early_stopping_rounds` or `verbose` to fit().",
        "• CatBoost categoricals: Pass `cat_features=cat_cols` to constructor. Do NOT label-encode before CatBoost.",
        "• XGBoost categoricals: Pass `enable_categorical=True` to constructor when using pd.Categorical features.",
        "• Pandas: `df['col'] = df['col'].fillna(x)` — NEVER `inplace=True` (broken in pandas 2.0+).",
        "• AdamW: `from torch.optim import AdamW` — NEVER `from transformers import AdamW`.",
        "• numpy/torch dtype: Ensure all features are numeric (float32/float64) before model.fit(). Cast with `.astype(float)` if needed.",
        "• After fillna(): verify `assert not df['col'].isna().any()` before calling .str/.apply on that column.",
        "• Val/test consistency: define a single `preprocess(df)` function called identically for both val and test — NEVER inline separate preprocessing blocks.",
        "• KeyError guard: NEVER access `test_df['target']` or any label column on test set — test CSVs have no target column.",
        "• IndexError guard: always check `len(X_train) > 0` and `len(X_val) > 0` before model.fit(); check `len(classes_) > 1` before using class indices.",
        "• TypeError guard: never pass `None` to sklearn metrics; always verify `y_pred is not None` and `len(y_pred) == len(y_val)` before scoring.",
        "• Column name safety: after any merge/join, verify expected columns exist with `assert 'col' in df.columns, f'Missing col, got {df.columns.tolist()}'`.",
        "",
        "🚫 **Execution Guidelines**:",
        "• NO tqdm (not installed), NO verbose=1",
        "• Print only 1 line per epoch (minimize logging)",
        "• Use DataLoader with num_workers>=2 for speed",
        "",
        "⚠️  **Self-Check Before Finalizing**:",
        "• Did predictions pass through model's learned weights during inference? (If NO → INVALID)",
        "• Did I generate submission.csv in correct path with ALL test predictions?",
        "• Did I print validation metric as the last line?",
        "• Did I use the COMPLETE training dataset (not a tiny subset)?",
        "• Did val and test inference use IDENTICAL preprocessing and postprocessing logic?",
        "",
        "💾 **OOF Predictions (for ensemble / stacking)**:",
        "• After k-fold training, save out-of-fold predictions: `np.save('./working/oof_preds.npy', oof_predictions)`",
        "• These enable downstream blending without re-running the model.",
        "",
        "🔧 **Post-Processing (squeeze extra performance)**:",
        "• Binary classification: optimize decision threshold on validation set (`threshold = np.percentile(val_probs, 100*(1-positive_rate))`)",
        "• Multiclass probabilities: apply temperature scaling if val ECE is high (`probs = softmax(logits / T)`)",
        "• Regression: clip predictions to observed training target range to prevent out-of-distribution outputs",
        "• Rank normalization before blending: `scipy.stats.rankdata(preds) / len(preds)`",
    ]
    if expose_prediction:
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

    if k_fold_validation > 1:
        impl_guideline.append(
            f"**Cross-Validation ({k_fold_validation}-fold StratifiedKFold required)**:\n"
            f"• Use StratifiedKFold(n_splits={k_fold_validation}, shuffle=True, random_state=42) for classification, KFold for regression.\n"
            f"• Train on each fold's train split, evaluate on its val split. Report mean CV score as the Final Validation Score.\n"
            f"• CRITICAL: Any group-level statistics (e.g. group mean of target, group size) MUST be computed inside each fold using only that fold's training rows. NEVER compute them on the full training set before splitting — this causes target leakage and inflated CV scores that do not generalize.\n"
            f"• For test predictions: retrain on the full training set (or average out-of-fold predictions) and generate submission.csv.\n"
            f"• Print: `print(f'Final Validation Score: {{mean_cv_score:.4f}} (std: {{std_cv_score:.4f}})')`"
        )

    return {"Implementation guideline": impl_guideline}
