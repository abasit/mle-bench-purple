"""Approach fingerprinting and equivalence class registry for graph-based search.

Implements the state-merging concept from Leurent & Maillard (2020)
"Monte-Carlo Graph Search: the Value of Merging Similar States".
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("MLEvolve")

# Pattern → tag mappings for code-based fingerprinting.
# Each tuple is (regex_pattern, tag_name).
_CODE_PATTERNS: list[tuple[str, str]] = [
    # Models — tree-based
    (r"(?:import xgboost|from xgboost|XGBClassifier|XGBRegressor)", "xgboost"),
    (r"(?:import lightgbm|from lightgbm|LGBMClassifier|LGBMRegressor)", "lightgbm"),
    (r"(?:import catboost|from catboost|CatBoostClassifier|CatBoostRegressor)", "catboost"),
    (r"RandomForestClassifier|RandomForestRegressor", "random_forest"),
    (r"GradientBoostingClassifier|GradientBoostingRegressor", "gradient_boosting"),
    (r"ExtraTreesClassifier|ExtraTreesRegressor", "extra_trees"),
    (r"AdaBoostClassifier|AdaBoostRegressor", "adaboost"),

    # Models — linear
    (r"LogisticRegression", "logistic_regression"),
    (r"LinearRegression|Ridge|Lasso|ElasticNet", "linear_model"),
    (r"(?:SVC|SVR|LinearSVC|LinearSVR)\b", "svm"),

    # Models — neural network
    (r"(?:import torch|from torch)", "pytorch"),
    (r"nn\.Module|nn\.Linear|nn\.Conv[12]d|nn\.LSTM|nn\.GRU|nn\.Transformer", "neural_net"),
    (r"(?:import tensorflow|from tensorflow|import keras|from keras)", "tensorflow"),
    (r"TabNetClassifier|TabNetRegressor", "tabnet"),

    # Models — neighbors / naive bayes
    (r"KNeighborsClassifier|KNeighborsRegressor", "knn"),
    (r"GaussianNB|MultinomialNB|BernoulliNB", "naive_bayes"),

    # Ensemble / stacking
    (r"StackingClassifier|StackingRegressor", "stacking"),
    (r"VotingClassifier|VotingRegressor", "voting"),
    (r"BaggingClassifier|BaggingRegressor", "bagging"),

    # Preprocessing
    (r"OneHotEncoder", "one_hot"),
    (r"LabelEncoder", "label_encoding"),
    (r"OrdinalEncoder", "ordinal_encoding"),
    (r"TargetEncoder|target_encode", "target_encoding"),
    (r"StandardScaler", "standard_scaling"),
    (r"MinMaxScaler", "minmax_scaling"),
    (r"RobustScaler", "robust_scaling"),
    (r"Normalizer", "normalizer"),
    (r"\bPCA\b", "pca"),
    (r"SimpleImputer|KNNImputer", "imputation"),
    (r"PolynomialFeatures", "polynomial_features"),
    (r"(?:import category_encoders|import ce\b)", "category_encoders"),

    # Feature selection
    (r"SelectKBest|SelectFromModel|RFE\b|RFECV", "feature_selection"),
    (r"mutual_info_classif|mutual_info_regression", "mutual_info"),

    # Validation
    (r"cross_val_score|cross_validate", "cross_validation"),
    (r"KFold|StratifiedKFold|RepeatedKFold|RepeatedStratifiedKFold", "kfold"),
    (r"GroupKFold|LeaveOneGroupOut", "group_kfold"),

    # Optimization
    (r"(?:import optuna|from optuna)", "optuna"),
    (r"(?:from hyperopt|import hyperopt)", "hyperopt"),
    (r"(?:from bayes_opt|BayesianOptimization)", "bayesian_opt"),
    (r"GridSearchCV|RandomizedSearchCV", "grid_search"),
    (r"BayesSearchCV", "bayes_search"),

    # Imbalanced
    (r"SMOTE|ADASYN|BorderlineSMOTE", "oversampling"),
    (r"RandomUnderSampler|TomekLinks|NearMiss", "undersampling"),

    # Deep learning specifics
    (r"(?:ResNet|resnet)", "resnet"),
    (r"(?:EfficientNet|efficientnet)", "efficientnet"),
    (r"(?:BERT|bert|BertModel|BertForSequence)", "bert"),
    (r"nn\.Embedding", "embeddings"),
    (r"(?:Dropout|nn\.Dropout)", "dropout"),
    (r"(?:BatchNorm|nn\.BatchNorm)", "batch_norm"),
    (r"(?:LayerNorm|nn\.LayerNorm)", "layer_norm"),
    (r"early_stopping|EarlyStopping", "early_stopping"),
    (r"(?:DataLoader|Dataset)\b", "dataloader"),

    # Misc techniques
    (r"(?:import shap|from shap)", "shap"),
    (r"(?:import autogluon|from autogluon|TabularPredictor)", "autogluon"),
]

# Additional keywords scanned in code_summary text (broader, less strict).
_SUMMARY_KEYWORDS: dict[str, list[str]] = {
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgbm", "light gbm"],
    "catboost": ["catboost"],
    "random_forest": ["random forest"],
    "neural_net": ["neural network", "neural net", "deep learning", "mlp"],
    "cnn": ["convolutional", "cnn", "conv2d"],
    "transformer": ["transformer", "attention mechanism"],
    "lstm": ["lstm", "recurrent", "gru"],
    "logistic_regression": ["logistic regression"],
    "svm": ["support vector", "svm"],
    "stacking": ["stacking", "stacked"],
    "ensemble": ["ensemble", "blending"],
    "one_hot": ["one-hot", "one hot", "onehot"],
    "target_encoding": ["target encoding", "target encode"],
    "label_encoding": ["label encoding", "label encode"],
    "standard_scaling": ["standard scal", "standardize"],
    "pca": ["pca", "principal component"],
    "cross_validation": ["cross-validation", "cross validation", "k-fold", "kfold"],
    "feature_engineering": ["feature engineering", "feature interaction"],
    "oversampling": ["smote", "oversampl"],
    "early_stopping": ["early stopping"],
    "dropout": ["dropout"],
    "batch_norm": ["batch norm"],
    "optuna": ["optuna"],
    "hyperparameter_tuning": ["hyperparameter", "tuning", "grid search", "bayesian opt"],
    "autogluon": ["autogluon"],
}

# Compile code patterns once.
_COMPILED_PATTERNS = [(re.compile(pat), tag) for pat, tag in _CODE_PATTERNS]


class ApproachFingerprint:
    """Extract a set of approach tags from code and/or code_summary."""

    @staticmethod
    def extract(code: str, code_summary: Optional[str] = None) -> frozenset[str]:
        tags: set[str] = set()

        # Scan code with regex patterns.
        if code:
            for pattern, tag in _COMPILED_PATTERNS:
                if pattern.search(code):
                    tags.add(tag)

        # Scan code_summary with keyword matching.
        if code_summary:
            summary_lower = code_summary.lower()
            for tag, keywords in _SUMMARY_KEYWORDS.items():
                for kw in keywords:
                    if kw in summary_lower:
                        tags.add(tag)
                        break

        return frozenset(tags)


@dataclass
class RegistrationResult:
    class_id: int
    similar_node_ids: list[str]
    is_near_duplicate: bool


class SimilarityRegistry:
    """Thread-safe registry tracking approach equivalence classes across all nodes.

    Two nodes are in the same equivalence class if their fingerprint Jaccard
    similarity exceeds `near_duplicate_threshold`. The registry enables:
    - Detecting redundant exploration across branches
    - Aggregating visit/reward statistics for class-level UCT
    - Providing approach landscape summaries for agent prompts
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        near_duplicate_threshold: float = 0.8,
    ):
        self.similarity_threshold = similarity_threshold
        self.near_duplicate_threshold = near_duplicate_threshold

        self._lock = threading.Lock()
        self._fingerprints: dict[str, frozenset[str]] = {}  # node_id → fingerprint
        self._node_branch: dict[str, int] = {}               # node_id → branch_id
        self._node_class: dict[str, int] = {}                # node_id → class_id
        self._classes: dict[int, set[str]] = {}              # class_id → {node_ids}
        self._class_fingerprint: dict[int, frozenset[str]] = {}  # class_id → representative fingerprint
        self._next_class_id: int = 0

    @staticmethod
    def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
        if not a and not b:
            return 0.0  # both unknown → not near-duplicates
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def register(
        self,
        node_id: str,
        code: str,
        code_summary: Optional[str],
        branch_id: int,
    ) -> RegistrationResult:
        fp = ApproachFingerprint.extract(code, code_summary)

        with self._lock:
            self._fingerprints[node_id] = fp
            self._node_branch[node_id] = branch_id

            # Remove from old class if this is a re-registration (e.g. code_summary update).
            old_class_id = self._node_class.get(node_id)
            if old_class_id is not None:
                self._classes[old_class_id].discard(node_id)
                # Clean up empty classes to avoid stale fingerprints attracting new nodes.
                if not self._classes[old_class_id]:
                    del self._classes[old_class_id]
                    del self._class_fingerprint[old_class_id]

            # Find best matching existing class.
            best_class_id = None
            best_sim = 0.0
            for class_id, class_fp in self._class_fingerprint.items():
                sim = self.jaccard_similarity(fp, class_fp)
                if sim > best_sim:
                    best_sim = sim
                    best_class_id = class_id

            is_near_duplicate = best_sim >= self.near_duplicate_threshold

            if is_near_duplicate and best_class_id is not None:
                # Join existing class.
                class_id = best_class_id
                self._classes[class_id].add(node_id)
                # Update class fingerprint to union (broadens the class).
                self._class_fingerprint[class_id] = self._class_fingerprint[class_id] | fp
            else:
                # Create new class.
                class_id = self._next_class_id
                self._next_class_id += 1
                self._classes[class_id] = {node_id}
                self._class_fingerprint[class_id] = fp

            self._node_class[node_id] = class_id
            similar_ids = [nid for nid in self._classes[class_id] if nid != node_id]

            if is_near_duplicate:
                logger.info(
                    f"[similarity] Node {node_id[:8]} joined class {class_id} "
                    f"(sim={best_sim:.2f}, class_size={len(self._classes[class_id])})"
                )

            return RegistrationResult(
                class_id=class_id,
                similar_node_ids=similar_ids,
                is_near_duplicate=is_near_duplicate,
            )

    def get_equivalence_class(self, node_id: str) -> list[str]:
        """All node IDs in the same equivalence class."""
        with self._lock:
            class_id = self._node_class.get(node_id)
            if class_id is None:
                return []
            return list(self._classes[class_id])

    def get_class_stats(self, node_id: str, journal) -> tuple[int, float]:
        """Aggregated (visits, total_reward) across the equivalence class."""
        with self._lock:
            class_id = self._node_class.get(node_id)
            if class_id is None:
                return 0, 0.0
            node_ids = set(self._classes[class_id])  # copy to avoid concurrent modification

        id2node = {n.id: n for n in journal.nodes}
        total_visits = 0
        total_reward = 0.0
        for nid in node_ids:
            node = id2node.get(nid)
            if node:
                total_visits += node.visits
                total_reward += node.total_reward
        return total_visits, total_reward

    def get_weighted_penalty_targets(self, node_id: str, journal) -> list[tuple]:
        """Returns [(eq_node, similarity)] for cross-branch penalty sharing.

        Thread-safe: all dict access is inside the lock.
        Only returns nodes from different branches.
        """
        with self._lock:
            class_id = self._node_class.get(node_id)
            if class_id is None:
                return []
            node_fp = self._fingerprints.get(node_id, frozenset())
            node_branch = self._node_branch.get(node_id)
            targets = []
            for eq_id in self._classes[class_id]:
                if eq_id == node_id:
                    continue
                eq_branch = self._node_branch.get(eq_id)
                if eq_branch != node_branch:
                    eq_fp = self._fingerprints.get(eq_id, frozenset())
                    sim = self.jaccard_similarity(node_fp, eq_fp)
                    targets.append((eq_id, sim))

        id2node = {n.id: n for n in journal.nodes}
        return [
            (id2node[eq_id], sim)
            for eq_id, sim in targets
            if eq_id in id2node
        ]

    def get_novelty_score(self, code: str, code_summary: Optional[str] = None) -> float:
        """0.0 = exact match to a large class, 1.0 = totally novel approach."""
        fp = ApproachFingerprint.extract(code, code_summary)
        if not fp:
            return 1.0

        with self._lock:
            best_sim = 0.0
            for class_fp in self._class_fingerprint.values():
                sim = self.jaccard_similarity(fp, class_fp)
                if sim > best_sim:
                    best_sim = sim

        return 1.0 - best_sim

    def get_approach_landscape(self) -> str:
        """Human-readable summary of explored approach clusters for prompt injection."""
        with self._lock:
            if not self._classes:
                return ""

            lines = []
            for class_id, node_ids in self._classes.items():
                fp = self._class_fingerprint[class_id]
                if not fp:
                    continue
                branches = set()
                for nid in node_ids:
                    bid = self._node_branch.get(nid)
                    if bid is not None:
                        branches.add(bid)
                tags_str = ", ".join(sorted(fp))
                lines.append(
                    f"Cluster {class_id} ({len(node_ids)} nodes, "
                    f"{len(branches)} branch{'es' if len(branches) != 1 else ''}): "
                    f"{tags_str}"
                )

        return "\n".join(lines)
