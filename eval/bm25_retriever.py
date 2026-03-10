"""BM25 Retriever for benchmark evaluation, using AutoDDG-compatible whitespace tokenization."""

import math
from collections import Counter
from typing import Dict, List, Tuple, Optional


class BM25Retriever:
    """Standard Okapi BM25 retriever with AutoDDG-compatible tokenization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Document length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b

        # Index state
        self.doc_ids: List[str] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.n_docs: int = 0
        self.doc_freqs: Counter = Counter()  # term -> number of docs containing it
        self._indexed = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization matching AutoDDG: text.lower().split()
        No stemming, no camelCase splitting, no special handling.
        """
        return text.lower().split()

    def index(self, documents: Dict[str, str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: {document_id: "text to index"}
        """
        self.doc_ids = []
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freqs = Counter()

        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            tf = Counter(tokens)

            self.doc_ids.append(doc_id)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))

            # Update document frequencies
            for term in tf:
                self.doc_freqs[term] += 1

        self.n_docs = len(self.doc_ids)
        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0.0
        self._indexed = True

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        tf_dict = self.doc_term_freqs[doc_idx]
        dl = self.doc_lengths[doc_idx]

        for term in query_tokens:
            if term not in tf_dict:
                continue

            tf = tf_dict[term]
            df = self.doc_freqs.get(term, 0)

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )

            score += idf * tf_norm

        return score

    def search(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """
        Search the index with a query.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of (document_id, score) tuples, sorted by score descending
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx in range(self.n_docs):
            score = self._bm25_score(query_tokens, idx)
            if score > 0:
                scores.append((self.doc_ids[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_search(
        self, queries: Dict[str, str], top_k: int = 1000
    ) -> Dict[str, List[str]]:
        """
        Search for multiple queries.

        Args:
            queries: {query_id: query_text}
            top_k: Number of top results per query

        Returns:
            {query_id: [ranked_document_ids]}
        """
        results = {}
        for qid, query_text in queries.items():
            ranked = self.search(query_text, top_k=top_k)
            results[qid] = [doc_id for doc_id, _ in ranked]
        return results

    def get_statistics(self) -> Dict[str, int]:
        """Return index statistics."""
        return {
            "n_docs": self.n_docs,
            "vocab_size": len(self.doc_freqs),
            "avg_doc_length": round(self.avg_dl, 1),
        }


def try_rank_bm25_retriever() -> Optional[type]:
    """
    Try to import rank_bm25 for potentially faster BM25.
    Falls back to our pure Python implementation above.
    """
    try:
        from rank_bm25 import BM25Okapi

        class RankBM25Retriever:
            """Wrapper around rank_bm25 library with AutoDDG tokenization."""

            def __init__(self, k1: float = 1.5, b: float = 0.75):
                self.k1 = k1
                self.b = b
                self.bm25 = None
                self.doc_ids: List[str] = []

            @staticmethod
            def _tokenize(text: str) -> List[str]:
                return text.lower().split()

            def index(self, documents: Dict[str, str]) -> None:
                self.doc_ids = list(documents.keys())
                corpus = [self._tokenize(documents[did]) for did in self.doc_ids]
                self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

            def search(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
                if self.bm25 is None:
                    raise RuntimeError("Index not built. Call index() first.")
                query_tokens = self._tokenize(query)
                scores = self.bm25.get_scores(query_tokens)
                ranked_indices = scores.argsort()[::-1][:top_k]
                return [
                    (self.doc_ids[idx], float(scores[idx]))
                    for idx in ranked_indices
                    if scores[idx] > 0
                ]

            def batch_search(
                self, queries: Dict[str, str], top_k: int = 1000
            ) -> Dict[str, List[str]]:
                results = {}
                for qid, query_text in queries.items():
                    ranked = self.search(query_text, top_k=top_k)
                    results[qid] = [doc_id for doc_id, _ in ranked]
                return results

            def get_statistics(self) -> Dict[str, int]:
                return {
                    "n_docs": len(self.doc_ids),
                    "backend": "rank_bm25",
                }

        return RankBM25Retriever
    except ImportError:
        return None


def create_bm25_retriever(k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    """
    Factory function: use rank_bm25 if available, otherwise pure Python fallback.
    """
    RankBM25Class = try_rank_bm25_retriever()
    if RankBM25Class:
        print("  [BM25] Using rank_bm25 backend")
        return RankBM25Class(k1=k1, b=b)
    else:
        print("  [BM25] Using pure Python BM25 implementation")
        return BM25Retriever(k1=k1, b=b)
