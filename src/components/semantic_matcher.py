from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar
import hashlib
import math

import numpy as np
from pathlib import Path

DEFAULT_LOCAL_MODEL_PATH = str(
    Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"
)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    sklearn_cosine_similarity = None
    _HAS_SKLEARN = False


T = TypeVar("T")


@dataclass(frozen=True)
class Match:
    """
    One matched pair between a previous item and a current item.
    """
    prev_index: int
    curr_index: int
    score: float


@dataclass(frozen=True)
class MatchResult(Generic[T]):
    """
    Result of greedy semantic matching between two item lists.
    """
    matches: list[Match]
    matched_prev_indices: set[int]
    matched_curr_indices: set[int]
    unmatched_prev_indices: list[int]
    unmatched_curr_indices: list[int]
    prev_items: list[T]
    curr_items: list[T]


class SemanticMatcher:
    """
    Semantic text matcher for free-form conflict / claim alignment.

    Design goals:
    - Prefer real embedding-based semantic similarity when sentence-transformers is available.
    - Fall back to TF-IDF cosine similarity when the environment does not provide an embedding model.
    - Keep the interface simple so transition_extractor.py can call it directly.
    - Support lightweight in-memory embedding caching.

    Typical use cases:
    - Match previous-round conflicts to current-round conflicts
    - Deduplicate semantically redundant claims inside a round
    - Optionally detect cross-round claim re-statements
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL_PATH,
        similarity_backend: str = "sentence_transformers",
        cache_embeddings: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Args:
            model_name:
                Sentence-transformers model name used when that backend is available.
                The default multilingual model is more practical here because your
                claims/conflicts may contain mixed Chinese/English expressions.
            similarity_backend:
                One of:
                - "auto": use sentence-transformers if available, otherwise TF-IDF
                - "sentence_transformers": force sentence-transformers
                - "tfidf": force TF-IDF cosine similarity
            cache_embeddings:
                Whether to cache text embeddings in memory.
            normalize_embeddings:
                Whether to L2-normalize dense embeddings before cosine similarity.
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.normalize_embeddings = normalize_embeddings

        self._embedding_cache: dict[str, np.ndarray] = {}
        self._model = None

        backend = similarity_backend.strip().lower()
        if backend not in {"auto", "sentence_transformers", "tfidf"}:
            raise ValueError(
                f"Unsupported similarity_backend={similarity_backend!r}. "
                f"Expected one of: 'auto', 'sentence_transformers', 'tfidf'."
            )

        if backend == "sentence_transformers":
            if not _HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is not installed, but "
                    "similarity_backend='sentence_transformers' was requested."
                )
            self.backend = "sentence_transformers"
        elif backend == "tfidf":
            if not _HAS_SKLEARN:
                raise ImportError(
                    "scikit-learn is not installed, but "
                    "similarity_backend='tfidf' was requested."
                )
            self.backend = "tfidf"
        else:
            if _HAS_SENTENCE_TRANSFORMERS:
                self.backend = "sentence_transformers"
            elif _HAS_SKLEARN:
                self.backend = "tfidf"
            else:
                raise ImportError(
                    "Neither sentence-transformers nor scikit-learn is available. "
                    "At least one similarity backend is required."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pairwise_similarity(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix between two text lists.

        Returns:
            A matrix of shape [len(texts_a), len(texts_b)].
            If either side is empty, returns an appropriately shaped zero matrix.
        """
        texts_a = [self._prepare_text(t) for t in texts_a]
        texts_b = [self._prepare_text(t) for t in texts_b]

        if len(texts_a) == 0 or len(texts_b) == 0:
            return np.zeros((len(texts_a), len(texts_b)), dtype=float)

        if self.backend == "sentence_transformers":
            emb_a = self._encode_dense(texts_a)
            emb_b = self._encode_dense(texts_b)
            return self._cosine_similarity_dense(emb_a, emb_b)

        if self.backend == "tfidf":
            return self._tfidf_cosine_similarity(texts_a, texts_b)

        raise RuntimeError(f"Unexpected backend: {self.backend}")

    def greedy_match_texts(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
        threshold: float,
    ) -> list[Match]:
        """
        Greedy bipartite matching on pairwise similarity matrix.

        Strategy:
        - Compute all pair scores
        - Repeatedly select the highest remaining pair
        - Accept only if score >= threshold
        - Remove the corresponding row and column
        - Continue until no valid pair remains

        This is simple, stable, and usually enough for conflict / claim alignment.

        Returns:
            List[Match], sorted by descending score then stable indices.
        """
        sim = self.pairwise_similarity(texts_a, texts_b)
        return self._greedy_match_from_similarity(sim=sim, threshold=threshold)

    def greedy_match_items(
        self,
        prev_items: Sequence[T],
        curr_items: Sequence[T],
        text_getter: Callable[[T], str],
        threshold: float,
    ) -> MatchResult[T]:
        """
        Greedy semantic matching between two item lists.

        Args:
            prev_items:
                Previous round items, e.g. previous conflicts.
            curr_items:
                Current round items, e.g. current conflicts.
            text_getter:
                Function extracting a free-form text field from each item.
                Example for conflicts: lambda c: c.conflict
                Example for claims:    lambda c: c.text
            threshold:
                Minimum cosine similarity to accept a match.

        Returns:
            MatchResult containing matches and unmatched indices.
        """
        prev_items_list = list(prev_items)
        curr_items_list = list(curr_items)

        prev_texts = [self._prepare_text(text_getter(item)) for item in prev_items_list]
        curr_texts = [self._prepare_text(text_getter(item)) for item in curr_items_list]

        matches = self.greedy_match_texts(prev_texts, curr_texts, threshold=threshold)

        matched_prev_indices = {m.prev_index for m in matches}
        matched_curr_indices = {m.curr_index for m in matches}

        unmatched_prev_indices = [
            i for i in range(len(prev_items_list)) if i not in matched_prev_indices
        ]
        unmatched_curr_indices = [
            j for j in range(len(curr_items_list)) if j not in matched_curr_indices
        ]

        return MatchResult(
            matches=matches,
            matched_prev_indices=matched_prev_indices,
            matched_curr_indices=matched_curr_indices,
            unmatched_prev_indices=unmatched_prev_indices,
            unmatched_curr_indices=unmatched_curr_indices,
            prev_items=prev_items_list,
            curr_items=curr_items_list,
        )

    # ------------------------------------------------------------------
    # Dense embedding backend
    # ------------------------------------------------------------------

    def _load_sentence_transformer(self) -> Any:
        if self._model is None:
            if not _HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers is not available.")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode_dense(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode texts into dense sentence embeddings, with optional caching.
        """
        model = self._load_sentence_transformer()

        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        results: list[np.ndarray | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if self.cache_embeddings and cache_key in self._embedding_cache:
                results[i] = self._embedding_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            encoded = model.encode(
                list(uncached_texts),
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)

            for local_idx, emb in enumerate(encoded):
                global_idx = uncached_indices[local_idx]
                text = texts[global_idx]
                cache_key = self._cache_key(text)
                if self.cache_embeddings:
                    self._embedding_cache[cache_key] = emb
                results[global_idx] = emb

        stacked = np.vstack([r for r in results if r is not None])

        if self.normalize_embeddings:
            # Some models may already normalize; this is harmless and stabilizes cosine.
            norms = np.linalg.norm(stacked, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            stacked = stacked / norms

        return stacked

    @staticmethod
    def _cosine_similarity_dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Cosine similarity for already-embedded matrices.
        Assumes rows are vectors.
        """
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=float)

        # If embeddings are normalized, dot product is cosine similarity.
        return np.clip(a @ b.T, -1.0, 1.0)

    # ------------------------------------------------------------------
    # TF-IDF backend
    # ------------------------------------------------------------------

    def _tfidf_cosine_similarity(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
    ) -> np.ndarray:
        """
        Fallback lexical-semantic similarity when sentence-transformers is unavailable.

        This is not truly semantic in the same way as dense embeddings, but it is still
        far more practical than exact normalized string matching and works without
        extra model dependencies.
        """
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is not available for TF-IDF fallback.")

        combined = list(texts_a) + list(texts_b)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=1,
        )
        mat = vectorizer.fit_transform(combined)
        mat_a = mat[: len(texts_a)]
        mat_b = mat[len(texts_a):]
        sim = sklearn_cosine_similarity(mat_a, mat_b)
        return np.asarray(sim, dtype=float)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy_match_from_similarity(
        sim: np.ndarray,
        threshold: float,
    ) -> list[Match]:
        """
        Greedy maximum matching from a similarity matrix.
        """
        if sim.size == 0:
            return []

        n_rows, n_cols = sim.shape
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        matches: list[Match] = []

        # Build sorted candidate list once.
        candidates: list[tuple[float, int, int]] = []
        for i in range(n_rows):
            for j in range(n_cols):
                score = float(sim[i, j])
                if math.isnan(score):
                    continue
                candidates.append((score, i, j))

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        for score, i, j in candidates:
            if score < threshold:
                break
            if i in used_rows or j in used_cols:
                continue

            used_rows.add(i)
            used_cols.add(j)
            matches.append(Match(prev_index=i, curr_index=j, score=score))

        return matches

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_text(text: str | None) -> str:
        """
        Lightweight text cleanup before embedding.
        Keep this intentionally simple:
        embedding models should handle free-form paraphrase variation,
        so we avoid brittle rule-heavy normalization here.
        """
        if text is None:
            return ""
        return " ".join(str(text).strip().split())

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()