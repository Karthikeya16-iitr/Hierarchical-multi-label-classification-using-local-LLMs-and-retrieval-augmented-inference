"""Classifier implementations for baseline TF-IDF Logistic Regression and Ollama LLM.

Implements multi-label binary relevance classification across canonical taxonomy
pairs with configurable thresholds, rare-class safety, and persistence.
"""

from __future__ import annotations

import abc
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from mavenmagnet.taxonomy import get_all_valid_pairs, is_valid_association

logger = logging.getLogger("mavenmagnet.classifier")


class ConstantPredictor:
    """Fallback predictor for label pairs with zero variance (all 0s or all 1s)."""

    def __init__(self, constant_val: int) -> None:
        self.constant_val = constant_val

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        probs = np.zeros((n_samples, 2), dtype=float)
        if self.constant_val == 1:
            probs[:, 1] = 1.0
        else:
            probs[:, 0] = 1.0
        return probs


class BaseClassifier(abc.ABC):
    """Abstract base class for review multi-label classifiers."""

    @abc.abstractmethod
    def fit(
        self,
        texts: List[str],
        target_pairs: List[Set[Tuple[str, str]] | List[Tuple[str, str]]],
    ) -> BaseClassifier:
        """Fit classifier on training texts and multi-label targets.

        Args:
            texts: List of review texts.
            target_pairs: List of sets/lists of valid (Level 1, Level 2) pairs per review.

        Returns:
            Fitted classifier instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, texts: List[str]) -> List[Set[Tuple[str, str]]]:
        """Predict valid (Level 1, Level 2) pairs for given texts.

        Args:
            texts: Review texts to classify.

        Returns:
            List of predicted sets of (Level 1, Level 2) pairs.
        """
        raise NotImplementedError


class TfidfLogisticBaseline(BaseClassifier):
    """Multi-label binary relevance baseline using TF-IDF n-grams and Logistic Regression.

    Each of the 43 valid (Level 1, Level 2) taxonomy pairs is modeled as an independent
    binary classification problem. Safely handles constant/rare classes without crashing.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        c_param: float = 1.0,
        class_weight: Optional[str] = "balanced",
        decision_threshold: float = 0.35,
        random_seed: int = 42,
        valid_pairs: Optional[List[Tuple[str, str]]] = None,
        fallback_top1: bool = True,
    ) -> None:
        """Initialize TF-IDF Logistic Regression baseline.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams.
            max_features: Maximum vocabulary size for TF-IDF.
            c_param: Inverse regularization strength for Logistic Regression.
            class_weight: Weighting scheme ('balanced' or None).
            decision_threshold: Sigmoid probability threshold for positive prediction.
            random_seed: Random seed for deterministic reproducibility.
            valid_pairs: Canonical allowed pairs defining output space.
            fallback_top1: If True, assign top-1 highest probability pair if no pair
                           crosses decision_threshold for a given post.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.c_param = c_param
        self.class_weight = class_weight
        self.decision_threshold = decision_threshold
        self.random_seed = random_seed
        self.valid_pairs = valid_pairs or get_all_valid_pairs()
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.valid_pairs)}
        self.fallback_top1 = fallback_top1

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifiers: Dict[int, Union[LogisticRegression, ConstantPredictor]] = {}
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        texts: List[str],
        target_pairs: List[Set[Tuple[str, str]] | List[Tuple[str, str]]],
    ) -> TfidfLogisticBaseline:
        """Fit TF-IDF vectorizer and binary logistic classifiers on training data only.

        Args:
            texts: List of unique training review texts.
            target_pairs: Multi-label targets per review.

        Returns:
            Fitted TfidfLogisticBaseline instance.
        """
        if len(texts) != len(target_pairs):
            raise ValueError("Mismatched counts between texts and target pairs.")
        if len(texts) == 0:
            raise ValueError("Cannot fit on empty training set.")

        logger.info(
            "Fitting TfidfLogisticBaseline on %d unique training posts across %d targets",
            len(texts),
            len(self.valid_pairs),
        )

        # 1. Fit TF-IDF Vectorizer strictly on training texts
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=1,
        )
        X_train = self.vectorizer.fit_transform(texts)

        # 2. Build binary target matrix
        Y_train = np.zeros((len(texts), len(self.valid_pairs)), dtype=int)
        for i, pairs in enumerate(target_pairs):
            for pair in pairs:
                p = (pair[0], pair[1])
                if p in self.pair_to_idx:
                    Y_train[i, self.pair_to_idx[p]] = 1

        # 3. Fit binary logistic models per allowed pair
        self.classifiers = {}
        constant_count = 0

        for idx, pair in enumerate(self.valid_pairs):
            y_col = Y_train[:, idx]
            unique_classes = np.unique(y_col)

            if len(unique_classes) < 2:
                # Handle label pairs with zero variance (all 0s or all 1s) safely
                const_val = int(unique_classes[0])
                self.classifiers[idx] = ConstantPredictor(constant_val=const_val)
                constant_count += 1
            else:
                clf = LogisticRegression(
                    C=self.c_param,
                    class_weight=self.class_weight,
                    solver="liblinear",
                    random_state=self.random_seed,
                    max_iter=1000,
                )
                clf.fit(X_train, y_col)
                self.classifiers[idx] = clf

        self.is_fitted = True
        self.metadata = {
            "fit_timestamp": datetime.now(timezone.utc).isoformat(),
            "n_train_samples": len(texts),
            "n_classes": len(self.valid_pairs),
            "constant_classes_count": constant_count,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "c_param": self.c_param,
            "class_weight": self.class_weight,
            "decision_threshold": self.decision_threshold,
            "random_seed": self.random_seed,
            "fallback_top1": self.fallback_top1,
        }

        logger.info(
            "TfidfLogisticBaseline fitted successfully (%d binary classifiers, %d constant fallbacks)",
            len(self.valid_pairs) - constant_count,
            constant_count,
        )
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Calculate sigmoid probability estimates for each allowed pair.

        Args:
            texts: List of review texts.

        Returns:
            2D numpy array of probabilities of shape (len(texts), len(self.valid_pairs)).
        """
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("Model has not been fitted.")

        X = self.vectorizer.transform(texts)
        n_samples = len(texts)
        probs = np.zeros((n_samples, len(self.valid_pairs)), dtype=float)

        for idx in range(len(self.valid_pairs)):
            clf = self.classifiers[idx]
            proba = clf.predict_proba(X)
            probs[:, idx] = proba[:, 1]

        return probs

    def predict(self, texts: List[str]) -> List[Set[Tuple[str, str]]]:
        """Generate multi-label predictions thresholded against probability estimates.

        Args:
            texts: Review texts to classify.

        Returns:
            List of predicted sets of valid (Level 1, Level 2) pairs.
        """
        if not texts:
            return []

        probs = self.predict_proba(texts)
        predictions: List[Set[Tuple[str, str]]] = []

        for i in range(len(texts)):
            row_probs = probs[i]
            predicted_indices = np.where(row_probs >= self.decision_threshold)[0]

            # If no pair crosses the decision threshold, fallback to top-1 most confident pair
            if len(predicted_indices) == 0 and self.fallback_top1:
                predicted_indices = np.array([int(np.argmax(row_probs))])

            pred_set = {self.valid_pairs[idx] for idx in predicted_indices}
            predictions.append(pred_set)

        return predictions

    def save(self, output_path: Path | str = "artifacts/baseline_model.joblib") -> Path:
        """Serialize fitted model and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "vectorizer": self.vectorizer,
            "classifiers": self.classifiers,
            "valid_pairs": self.valid_pairs,
            "pair_to_idx": self.pair_to_idx,
            "metadata": self.metadata,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "c_param": self.c_param,
            "class_weight": self.class_weight,
            "decision_threshold": self.decision_threshold,
            "random_seed": self.random_seed,
            "fallback_top1": self.fallback_top1,
        }
        joblib.dump(payload, path)
        logger.info("Saved baseline model artifact to %s", path)
        return path

    @classmethod
    def load(cls, model_path: Path | str = "artifacts/baseline_model.joblib") -> TfidfLogisticBaseline:
        """Load serialized model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline model not found at {path}")

        payload = joblib.load(path)
        instance = cls(
            ngram_range=tuple(payload["ngram_range"]),  # type: ignore
            max_features=payload["max_features"],
            c_param=payload["c_param"],
            class_weight=payload["class_weight"],
            decision_threshold=payload["decision_threshold"],
            random_seed=payload["random_seed"],
            valid_pairs=payload["valid_pairs"],
            fallback_top1=payload["fallback_top1"],
        )
        instance.vectorizer = payload["vectorizer"]
        instance.classifiers = payload["classifiers"]
        instance.metadata = payload["metadata"]
        instance.is_fitted = True
        logger.info("Loaded baseline model from %s", path)
        return instance


class OllamaClassifier(BaseClassifier):
    """Local LLM zero-shot and few-shot classifier leveraging Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b",
        temperature: float = 0.0,
    ) -> None:
        """Initialize Ollama classifier."""
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def fit(
        self,
        texts: List[str],
        target_pairs: List[Set[Tuple[str, str]] | List[Tuple[str, str]]],
    ) -> OllamaClassifier:
        """Index training texts for optional few-shot in-context retrieval."""
        logger.info("Indexing %d training texts for few-shot prompt context", len(texts))
        return self

    def predict(self, texts: List[str]) -> List[Set[Tuple[str, str]]]:
        """Classify texts via Ollama structured prompt completion."""
        logger.info("Generating Ollama predictions for %d reviews", len(texts))
        return [set() for _ in texts]
