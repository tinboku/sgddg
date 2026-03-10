"""Tests for AliasStore."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.alias_store import AliasStore


def test_add_and_retrieve_aliases(db_connection):
    store = AliasStore(db_connection)
    aliases = [
        {"alias_text": "Revenue", "concept_id": "revenue"},
        {"alias_text": "Sales", "concept_id": "revenue"},
    ]
    store.add_aliases_batch(aliases)
    all_aliases = store.get_all_aliases()
    assert len(all_aliases) == 2


def test_get_statistics(db_connection):
    store = AliasStore(db_connection)
    store.add_aliases_batch([{"alias_text": "Rev", "concept_id": "r"}])
    stats = store.get_statistics()
    assert stats["total_aliases"] == 1


def test_duplicate_alias_ignored(db_connection):
    store = AliasStore(db_connection)
    alias = {"alias_text": "Revenue", "concept_id": "revenue"}
    store.add_aliases_batch([alias, alias])
    stats = store.get_statistics()
    assert stats["total_aliases"] == 1
