"""BM25 Index - sparse lexical retrieval over KG concepts for concept matching."""

import math
import re
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class BM25Index:
    """BM25 (Okapi) sparse retrieval index over KG concept names, definitions, and aliases."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.avg_dl: float = 0.0
        self.n_docs: int = 0
        self._built = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + underscore tokenizer with lowercasing."""
        if not text:
            return []
        # Replace underscores, hyphens, camelCase splits with spaces
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace("_", " ").replace("-", " ")
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def build_from_concepts(self, concepts: List[Dict[str, Any]]):
        """
        Build BM25 index from a list of KG concepts.

        Args:
            concepts: List of concept dicts with id, display_name, definition, aliases, etc.
        """
        self.documents = concepts
        self.doc_tokens = []
        self.doc_freqs = {}

        for concept in concepts:
            # Combine display_name, definition, and aliases into a single document
            parts = [
                concept.get("display_name", ""),
                concept.get("definition", ""),
            ]
            aliases = concept.get("aliases", [])
            if isinstance(aliases, list):
                parts.extend(aliases)

            full_text = " ".join(str(p) for p in parts if p)
            tokens = self._tokenize(full_text)
            self.doc_tokens.append(tokens)

            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.n_docs = len(self.documents)
        total_tokens = sum(len(t) for t in self.doc_tokens)
        self.avg_dl = total_tokens / self.n_docs if self.n_docs > 0 else 0.0
        self._built = True

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.doc_tokens[doc_idx]
        dl = len(doc_tokens)
        tf_counter = Counter(doc_tokens)

        score = 0.0
        for qt in query_tokens:
            if qt not in self.doc_freqs:
                continue

            tf = tf_counter.get(qt, 0)
            if tf == 0:
                continue

            df = self.doc_freqs[qt]
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )
            score += idf * tf_norm

        return score

    def search(
        self, query: str, top_k: int = 10, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            threshold: Minimum BM25 score threshold.

        Returns:
            List of dicts with concept info and BM25 scores.
        """
        if not self._built or self.n_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Score all documents
        scored = []
        for idx in range(self.n_docs):
            score = self._bm25_score(query_tokens, idx)
            if score > threshold:
                scored.append((idx, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for idx, score in scored[:top_k]:
            concept = self.documents[idx]
            results.append({
                "id": concept.get("id", concept.get("concept_id", "unknown")),
                "display_name": concept.get("display_name", ""),
                "definition": concept.get("definition", ""),
                "score": score,
                "match_method": "bm25",
                "metadata": concept,
            })

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "n_docs": self.n_docs,
            "vocab_size": len(self.doc_freqs),
            "avg_doc_length": self.avg_dl,
            "built": self._built,
        }
