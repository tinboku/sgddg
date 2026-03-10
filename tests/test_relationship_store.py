"""Tests for RelationshipStore."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.relationship_store import RelationshipStore


def test_add_and_find_parents(db_connection):
    store = RelationshipStore(db_connection)
    store.add_relationships_batch([{
        "source_concept_id": "revenue",
        "relationship_type": "is_a",
        "target_concept_id": "financial_metric",
        "confidence": 0.9,
        "evidence": "test",
    }])
    parents = store.find_parents("revenue")
    assert parents == ["financial_metric"]


def test_find_children(db_connection):
    store = RelationshipStore(db_connection)
    store.add_relationships_batch([{
        "source_concept_id": "revenue",
        "relationship_type": "is_a",
        "target_concept_id": "financial_metric",
    }])
    children = store.find_children("financial_metric")
    assert children == ["revenue"]


def test_get_statistics(db_connection):
    store = RelationshipStore(db_connection)
    store.add_relationships_batch([
        {"source_concept_id": "a", "relationship_type": "is_a", "target_concept_id": "b"},
        {"source_concept_id": "c", "relationship_type": "related_to", "target_concept_id": "d"},
    ])
    stats = store.get_statistics()
    assert stats["total_relationships"] == 2
