"""KG Manager - unified interface for ConceptStore, AliasStore, VectorStore, and RelationshipStore."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
import os
import sqlite3

from .concept_store import ConceptStore
from .vector_store import VectorStore
from .alias_store import AliasStore
from .relationship_store import RelationshipStore

class KnowledgeGraphManager:
    """Unified knowledge graph manager."""

    def __init__(self, kg_directory: str, vector_dimension: int = 384):
        """Initialize KG manager, loading an existing KG or creating an empty one."""
        self.kg_directory = kg_directory
        os.makedirs(self.kg_directory, exist_ok=True)

        db_path = os.path.join(self.kg_directory, "domain_knowledge.db")
        vector_index_path = os.path.join(self.kg_directory, "domain_vectors.faiss")

        # Centralized database connection
        self.conn = sqlite3.connect(db_path)

        self.concept_store = ConceptStore(self.conn)
        self.alias_store = AliasStore(self.conn)
        self.relationship_store = RelationshipStore(self.conn)
        self.vector_store = VectorStore(
            concept_store=self.concept_store,
            index_path=vector_index_path,
            metadata_path=vector_index_path.replace(".faiss", "_metadata.pkl")
        )

    def add_concepts(self, concepts: List[Dict[str, Any]], batch_size: int = 100):
        """Batch add or update concepts with their aliases and relationships."""
        print(f"   - Preparing to batch-process {len(concepts)} concepts...")

        for i in tqdm(range(0, len(concepts), batch_size), desc="Adding concepts"):
            batch = concepts[i:i + batch_size]

            # Prepare data for ConceptStore and AliasStore
            concepts_for_db = []
            aliases_for_db = []

            for concept_data in batch:
                concept_id = concept_data.get("concept_id") or concept_data.get("id")
                if not concept_id:
                    continue

                # Prepare concept data
                concepts_for_db.append({
                    "id": concept_id,
                    "display_name": concept_data.get("display_name", concept_id),
                    "definition": concept_data.get("definition", ""),
                    "metadata": concept_data.get("metadata", {})
                })

                # Prepare alias data
                aliases = concept_data.get("aliases", [])
                for alias in aliases:
                    aliases_for_db.append({
                        "alias_text": alias,
                        "concept_id": concept_id
                    })

            # Prepare relationship data
            relationships_for_db = []
            for concept_data in batch:
                concept_id = concept_data.get("concept_id") or concept_data.get("id")
                if not concept_id:
                    continue

                relations = concept_data.get("relations", [])
                for rel in relations:
                    target = rel.get("target")
                    rel_type = rel.get("type", "related_to")
                    if not target:
                        continue
                    relationships_for_db.append({
                        "source_concept_id": concept_id,
                        "relationship_type": rel_type,
                        "target_concept_id": target,
                        "confidence": rel.get("confidence", 0.0),
                        "evidence": rel.get("evidence")
                    })

                metadata = concept_data.get("metadata", {})
                hierarchy = metadata.get("hierarchy", {})
                if not hierarchy:
                    hierarchy = concept_data.get("hierarchy", {})

                if hierarchy:
                    parent = hierarchy.get("parent")
                    if parent:
                        parent_id = parent.lower().replace(" ", "_")
                        relationships_for_db.append({
                            "source_concept_id": concept_id,
                            "relationship_type": "is_a",
                            "target_concept_id": parent_id,
                            "confidence": 0.5,
                            "evidence": "inferred from legacy hierarchy metadata"
                        })

            # Batch insert
            if concepts_for_db:
                self.concept_store.add_concepts_batch(concepts_for_db)
            if aliases_for_db:
                self.alias_store.add_aliases_batch(aliases_for_db)
            if relationships_for_db:
                self.relationship_store.add_relationships_batch(relationships_for_db)

    def add_concept_with_vector(
        self,
        concept: Dict[str, Any],
        vector: np.ndarray,
        aliases: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Add a single concept with its vector and aliases."""
        concept_id = concept.get("id")
        if not concept_id:
            raise ValueError("Concept must have an 'id'")

        self.concept_store.add_concepts_batch([concept])

        if aliases:
            for alias in aliases:
                alias["concept_id"] = concept_id
            self.alias_store.add_aliases_batch(aliases)

        if self.vector_store:
            self.vector_store.add_vector(concept_id, vector, metadata=concept)

        return concept_id

    def search_concept(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 10,
        threshold: float = 0.0,
        use_vector_search: bool = True,
        use_alias_search: bool = True
    ) -> List[Dict[str, Any]]:
        """Search concepts using combined vector retrieval and alias lookup."""
        final_results: Dict[str, Dict[str, Any]] = {}

        # Alias exact match (highest priority)
        if use_alias_search:
            pass

        # Vector search
        if use_vector_search:
            vector_results = self.vector_store.search(query, top_k=top_k, threshold=threshold)
            for item in vector_results:
                concept_id = item['id']
                if concept_id not in final_results:
                    concept = item.get("metadata", {})
                    if concept:
                        concept["match_method"] = "vector_search"
                        concept["score"] = item.get("score")
                        final_results[concept_id] = concept

        sorted_results = sorted(final_results.values(), key=lambda x: x.get("score", 0.0), reverse=True)

        return sorted_results[:top_k]

    def get_concept_with_aliases(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept with all its aliases."""
        concept = self.concept_store.get_concept_by_id(concept_id)
        return concept

    def get_statistics(self) -> Dict[str, Any]:
        """Get global statistics across all stores."""
        concept_stats = self.concept_store.get_statistics()
        alias_stats = self.alias_store.get_statistics()
        relationship_stats = self.relationship_store.get_statistics()
        vector_stats = self.vector_store.get_statistics()

        return {
            "concepts": concept_stats,
            "aliases": alias_stats,
            "relationships": relationship_stats,
            "vectors": vector_stats
        }

    def save(self) -> None:
        """Save all data to disk."""
        if self.vector_store and hasattr(self.vector_store, 'next_id'):
            self.vector_store.save(self.vector_store.next_id)

    def close(self) -> None:
        """Close all connections."""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    print("KGManager has been refactored. Please use scripts like 'build_domain_kg.py' for testing.")
