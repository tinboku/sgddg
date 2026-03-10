"""Tests for ConceptStore."""
import sys
from pathlib import Path

# Allow running tests from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.concept_store import ConceptStore


def test_add_and_retrieve_concept(db_connection):
    store = ConceptStore(db_connection)
    concept = {
        "id": "test_revenue",
        "display_name": "Revenue",
        "definition": "Total income",
    }
    store.add_concepts_batch([concept])
    retrieved = store.get_concept_by_id("test_revenue")
    assert retrieved is not None
    assert retrieved["display_name"] == "Revenue"


def test_get_all_concepts(db_connection):
    store = ConceptStore(db_connection)
    concepts = [
        {"id": "a", "display_name": "A", "definition": "Def A"},
        {"id": "b", "display_name": "B", "definition": "Def B"},
    ]
    store.add_concepts_batch(concepts)
    all_concepts = store.get_all_concepts()
    assert len(all_concepts) == 2


def test_get_statistics(db_connection):
    store = ConceptStore(db_connection)
    store.add_concepts_batch([{"id": "x", "display_name": "X", "definition": ""}])
    stats = store.get_statistics()
    assert stats["total_concepts"] == 1


def test_upsert_concept(db_connection):
    store = ConceptStore(db_connection)
    store.add_concepts_batch([{"id": "c", "display_name": "Old", "definition": ""}])
    store.add_concepts_batch([{"id": "c", "display_name": "New", "definition": "Updated"}])
    retrieved = store.get_concept_by_id("c")
    assert retrieved["display_name"] == "New"


def test_get_nonexistent_concept(db_connection):
    store = ConceptStore(db_connection)
    assert store.get_concept_by_id("nonexistent") is None
