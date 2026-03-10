"""Vector Store - manages FAISS-based vector index for semantic concept retrieval."""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed, vector retrieval unavailable")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed, cannot auto-generate vectors")

from .concept_store import ConceptStore


class VectorStore:
    """FAISS + SentenceTransformer vector store with BGE instruction prefix support."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    BGE_MODEL_PREFIXES = {"bge-large", "bge-base", "bge-small", "bge-m3"}

    def __init__(self, concept_store: 'ConceptStore', index_path: str, metadata_path: str, model_name: str = 'BAAI/bge-large-en-v1.5'):
        """Initialize vector store with BGE model support."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is not installed. Please install it to use VectorStore.")

        self.concept_store = concept_store
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._is_bge = any(prefix in model_name.lower() for prefix in self.BGE_MODEL_PREFIXES)
        self.index = None
        self.id_to_metadata = {}
        self.concept_id_to_idx = {}
        self.next_id = 0
        self.load()

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts into vectors (adds BGE prefix for queries)."""
        if not self.model:
            raise RuntimeError("SentenceTransformer model is not available.")
        if self._is_bge and is_query:
            texts = [self.BGE_QUERY_PREFIX + t for t in texts]
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Public API: encode texts into vectors."""
        return self._encode(texts, is_query=is_query)

    def add_vector(
        self,
        concept_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a single vector to the index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed")

        # Normalize for cosine similarity
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        vector = vector.astype('float32')
        faiss.normalize_L2(vector)

        self.index.add(vector)

        # Save metadata
        idx = self.next_id
        self.id_to_metadata[idx] = metadata or {}
        self.id_to_metadata[idx]["concept_id"] = concept_id
        self.concept_id_to_idx[concept_id] = idx
        self.next_id += 1

        return idx

    def add_batch(
        self,
        concept_ids: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """Batch add vectors to the index."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed")

        # Normalize
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        # Save metadata
        indices = []
        for i, concept_id in enumerate(concept_ids):
            idx = self.next_id
            metadata = metadata_list[i] if metadata_list else {}
            metadata["concept_id"] = concept_id
            self.id_to_metadata[idx] = metadata
            self.concept_id_to_idx[concept_id] = idx
            indices.append(idx)
            self.next_id += 1

        return indices

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search by string query or vector, returning top-K results above threshold."""
        if not FAISS_AVAILABLE or self.index.ntotal == 0:
            return []

        # Encode string queries
        if isinstance(query, str):
            if not self.model:
                print("Warning: SentenceTransformer model is not available for encoding.")
                return []
            query_vector = self._encode([query], is_query=True)
        else:
            query_vector = query

        # Normalize query vector
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < threshold:
                continue

            if idx == -1:
                continue

            metadata = self.id_to_metadata.get(int(idx), {})
            concept_id = metadata.get("concept_id", "unknown")
            results.append({
                "id": concept_id,
                "score": float(score),
                "metadata": metadata
            })

        return results

    def get_vector(self, concept_id: str) -> Optional[np.ndarray]:
        """Get vector for a concept, or None if not found."""
        if not FAISS_AVAILABLE:
            return None

        idx = self.concept_id_to_idx.get(concept_id)
        if idx is None:
            return None

        return self.index.reconstruct(idx)

    def create_and_save_index(self, rebuild: bool = False):
        """Build FAISS index from all concepts in ConceptStore."""
        if self.index is not None and not rebuild:
            print("   - Vector index already exists, skipping rebuild.")
            return

        if not self.model:
            print("   - Warning: SentenceTransformer not installed, cannot build vector index.")
            return

        print("   - Fetching all concepts from ConceptStore...")
        concepts = self.concept_store.get_all_concepts()

        if not concepts:
            print("   - Warning: No concepts in ConceptStore, skipping index build.")
            return

        print(f"   - Found {len(concepts)} concepts to index.")

        # Prepare texts, IDs, and metadata
        texts_to_encode = []
        concept_ids = []
        metadata_list = []

        for concept in concepts:
            text = f"{concept.get('display_name', '')}. {concept.get('definition', '')}"
            texts_to_encode.append(text.strip())
            concept_ids.append(concept.get("id"))
            metadata_list.append(concept)

        # Batch encode
        print("   - Encoding concepts to vectors...")
        vectors = self._encode(texts_to_encode)

        # Initialize new FAISS index
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.id_to_metadata = {}
        next_id = 0

        # Batch add to index
        print("   - Adding vectors to FAISS index...")
        if vectors.shape[0] > 0:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)

            for i, concept_id in enumerate(concept_ids):
                self.id_to_metadata[i] = metadata_list[i]

            next_id = len(concept_ids)

        # Save
        print("   - Saving index and metadata...")
        self.save(next_id)
        print(f"   - Successfully built and saved vector index for {len(concepts)} concepts.")

    def save(self, next_id: int) -> None:
        """Save index and metadata to disk."""
        if not FAISS_AVAILABLE:
            return

        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                "id_to_metadata": self.id_to_metadata,
                "next_id": next_id
            }, f)

    def load(self) -> bool:
        """Load index and metadata from disk."""
        if not FAISS_AVAILABLE:
            return False

        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            dimension = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(dimension)
            self.id_to_metadata = {}
            return False

        try:
            self.index = faiss.read_index(self.index_path)

            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.id_to_metadata = data["id_to_metadata"]
                self.next_id = data.get("next_id", len(self.id_to_metadata))
                self.concept_id_to_idx = {
                    meta.get("concept_id"): idx for idx, meta in self.id_to_metadata.items()
                }

            return True
        except Exception as e:
            print(f"   - Warning: Failed to load vector index: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not FAISS_AVAILABLE:
            return {"total_vectors": 0, "dimension": self.dimension}

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "FAISS-IndexFlatIP"
        }


if __name__ == "__main__":
    import sqlite3
    import tempfile

    print("VectorStore Demo")

    if not FAISS_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        print("FAISS or sentence-transformers not installed, skipping demo")
        if not FAISS_AVAILABLE: print("  Install FAISS: pip install faiss-cpu")
        if not TRANSFORMERS_AVAILABLE: print("  Install: pip install sentence-transformers")
        exit(1)

    conn = sqlite3.connect(":memory:")
    cs = ConceptStore(conn)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(
            concept_store=cs,
            index_path=os.path.join(tmpdir, "demo_vector.faiss"),
            metadata_path=os.path.join(tmpdir, "demo_vector_metadata.pkl")
        )

        concepts_to_add = {
            "revenue": ("Total Revenue", "Total income from sales"),
            "profit": ("Net Profit", "The residual income after all expenses"),
            "age": ("Patient Age", "The age of the patient in years")
        }

        ids = list(concepts_to_add.keys())
        texts = [" ".join(v) for v in concepts_to_add.values()]
        vectors = store.encode(texts)

        store.add_batch(ids, vectors, [{"display_name": concepts_to_add[cid][0]} for cid in ids])
        print(f"Added {len(ids)} vectors")

        query_text = "company earnings"
        results = store.search(query_text, top_k=3)
        print(f"\nSearch '{query_text}' (Top-3):")
        for result in results:
            print(f"  - {result['id']}: {result['score']:.4f} ({result['metadata'].get('display_name')})")

        store.save(store.next_id)
        print(f"\nIndex saved to: {store.index_path}")

        stats = store.get_statistics()
        print(f"Stats: {stats}")

    conn.close()
    print("Demo complete!")
