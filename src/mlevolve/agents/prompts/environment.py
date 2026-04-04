"""Environment/package prompt."""


def get_prompt_environment():
    """Installed packages description."""
    return {
        "Installed Packages": (
            "The following packages are pre-installed and available for use. "
            "Do NOT attempt to pip install anything — use only what is listed here.\n\n"
            "**Core ML**: scikit-learn, xgboost (`import xgboost`), lightgbm (`import lightgbm`), "
            "catboost (`import catboost`), pytorch-tabnet (`from pytorch_tabnet`), "
            "autogluon (`from autogluon.tabular import TabularPredictor`)\n"
            "**Deep Learning**: torch, torchvision, torchaudio, transformers, timm, "
            "accelerate, pytorch-lightning, einops, peft, sentence-transformers\n"
            "**Optimization**: optuna (`import optuna`), hyperopt (`from hyperopt import fmin, hp`), "
            "bayesian-optimization (`from bayes_opt import BayesianOptimization`), "
            "scikit-optimize (`from skopt import BayesSearchCV`)\n"
            "**Data**: numpy, pandas, polars, scipy, pyarrow, fastparquet, h5py, openpyxl\n"
            "**Preprocessing**: imbalanced-learn (`from imblearn.over_sampling import SMOTE`), "
            "category_encoders (`import category_encoders as ce`), sklearn-pandas\n"
            "**Vision**: opencv-python-headless (`import cv2`), Pillow, albumentations, "
            "torchvision, segmentation_models_pytorch, ultralytics\n"
            "**NLP**: spacy, nltk, gensim, fasttext\n"
            "**Time Series**: sktime, statsforecast, statsmodels\n"
            "**Tabular**: shap, lightgbm, xgboost, catboost\n"
            "**Visualization**: matplotlib, seaborn, plotly\n"
            "**Utilities**: tqdm, joblib, numba\n\n"
            "For neural networks, use PyTorch rather than TensorFlow."
        )
    }
