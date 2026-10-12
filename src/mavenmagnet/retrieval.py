"""TF-IDF cosine similarity retrieval system over unique grouped training posts.

Retrieves semantically similar few-shot exemplars with complete label sets
while strictly preventing evaluation data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mavenmagnet.dataset_builder import GroupedPostRecord, PostRecord
from mavenmagnet.preprocessing import clean_text

logger = logging.getLogger("mavenmagnet.retrieval")


@dataclass(frozen=True)
class RetrievedExemplar:
    """Individual retrieved exemplar containing similarity score and complete label set."""

    post_id: str
    original_text: str
    clean_text: str
    pairs: List[Tuple[str, str]]
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert exemplar to dictionary."""
        return {
            "post_id": self.post_id,
            "original_text": self.original_text,
            "clean_text": self.clean_text,
            "pairs": [[p[0], p[1]] for p in self.pairs],
            "similarity_score": round(self.similarity_score, 4),
        }


class ExemplarRetriever:
    """TF-IDF cosine similarity index over unique grouped posts.

    CRITICAL LEAKAGE REQUIREMENT:
    When evaluating on validation or test sets, this retriever MUST be indexed
    exclusively on the training partition. It supports explicit post ID / text
    exclusion to strictly guarantee that no validation post or label is ever retrieved.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        sublinear_tf: bool = True,
        train_records: Optional[List[GroupedPostRecord]] = None,
    ) -> None:
        """Initialize retriever.

        Args:
            ngram_range: n-gram range for TF-IDF feature extraction.
            max_features: Maximum vocabulary size.
            sublinear_tf: Apply sublinear scaling to term frequencies.
            train_records: Optional records to fit immediately on initialization.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.index_matrix: Optional[Any] = None
        self.records: List[GroupedPostRecord] = []
        self.is_fitted = False

        if train_records:
            self.fit(train_records)

    def fit(self, records: List[GroupedPostRecord]) -> ExemplarRetriever:
        """Index unique training posts and their complete label sets.

        Args:
            records: List of unique grouped training post records.

        Returns:
            Fitted ExemplarRetriever instance.
        """
        if not records:
            logger.warning("Attempted to fit ExemplarRetriever on empty records.")
            self.records = []
            self.is_fitted = False
            return self

        self.records = list(records)
        corpus = [r.clean_text for r in self.records]

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            min_df=1,
        )
        self.index_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

        logger.info(
            "Fitted ExemplarRetriever on %d unique posts (vocab size: %d)",
            len(self.records),
            len(self.vectorizer.vocabulary_),
        )
        return self

    def retrieve(
        self,
        query_text: str,
        k: int = 3,
        exclude_post_ids: Optional[Set[str]] = None,
        exclude_texts: Optional[Set[str]] = None,
        min_similarity: float = 0.0,
    ) -> List[RetrievedExemplar]:
        """Retrieve top-k most similar exemplars ranked by TF-IDF cosine similarity.

        Guarantees deterministic ranking, zero duplicates, and strict leakage prevention.

        Args:
            query_text: Unlabeled query review string.
            k: Number of exemplars to retrieve (typically 3 to 5).
            exclude_post_ids: Set of post IDs to strictly exclude from results.
            exclude_texts: Set of cleaned texts to strictly exclude from results.
            min_similarity: Minimum cosine similarity score threshold.

        Returns:
            Sorted list of top-k RetrievedExemplar instances.
        """
        if not self.is_fitted or not self.records or self.vectorizer is None or self.index_matrix is None:
            return []

        cleaned_query = clean_text(query_text)
        if not cleaned_query:
            return []

        exclude_ids = exclude_post_ids or set()
        exclude_txt = {clean_text(t) for t in (exclude_texts or set())}

        # Vectorize query
        q_vec = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(q_vec, self.index_matrix)[0]

        # Candidate selection and tie-breaking:
        # Sort by (-score, post_id) for strictly deterministic ranking
        candidates: List[Tuple[float, str, GroupedPostRecord]] = []
        for idx, record in enumerate(self.records):
            if record.post_id in exclude_ids or record.clean_text in exclude_txt:
                continue

            score = float(scores[idx])
            if score >= min_similarity:
                candidates.append((score, record.post_id, record))

        # Sort descending by score, ascending by post_id
        candidates.sort(key=lambda x: (-x[0], x[1]))

        # Deduplicate and return top k
        results: List[RetrievedExemplar] = []
        seen_post_ids: Set[str] = set()

        for score, post_id, record in candidates:
            if post_id in seen_post_ids:
                continue
            seen_post_ids.add(post_id)

            results.append(
                RetrievedExemplar(
                    post_id=record.post_id,
                    original_text=record.original_text,
                    clean_text=record.clean_text,
                    pairs=record.pairs,
                    similarity_score=score,
                )
            )
            if len(results) >= k:
                break

        return results

    def retrieve_exemplars(self, query_text: str, k: int = 3) -> List[PostRecord]:
        """Backward-compatible retrieval method returning PostRecord instances."""
        exemplars = self.retrieve(query_text, k=k)
        return [
            PostRecord(
                post_id=ex.post_id,
                original_text=ex.original_text,
                clean_text=ex.clean_text,
                pairs=ex.pairs,
                original_row_indices=[],
            )
            for ex in exemplars
        ]

    def save(self, output_path: Path | str = "artifacts/retriever.joblib") -> Path:
        """Serialize retriever index to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted retriever.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "vectorizer": self.vectorizer,
            "index_matrix": self.index_matrix,
            "records": [r.to_dict() for r in self.records],
        }
        joblib.dump(payload, path)
        logger.info("Saved retriever artifact to %s", path)
        return path

    @classmethod
    def load(cls, model_path: Path | str = "artifacts/retriever.joblib") -> ExemplarRetriever:
        """Load serialized retriever from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Retriever artifact not found at {path}")

        payload = joblib.load(path)
        instance = cls(
            ngram_range=tuple(payload["ngram_range"]),  # type: ignore
            max_features=payload["max_features"],
            sublinear_tf=payload["sublinear_tf"],
        )
        instance.vectorizer = payload["vectorizer"]
        instance.index_matrix = payload["index_matrix"]
        instance.records = [GroupedPostRecord.from_dict(d) for d in payload["records"]]
        instance.is_fitted = True
        logger.info("Loaded retriever from %s with %d records", path, len(instance.records))
        return instance
