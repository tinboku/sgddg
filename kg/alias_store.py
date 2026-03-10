"""Alias Store - manages concept aliases, synonyms, and multilingual variants."""

import sqlite3
from typing import List, Dict, Any

class AliasStore:
    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create the aliases table."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS aliases (
                alias TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                PRIMARY KEY (alias, concept_id)
            )
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias)")
        self.conn.commit()

    def add_aliases_batch(self, aliases: List[Dict[str, str]]):
        """Batch insert aliases."""
        if not aliases:
            return

        records = [(alias["alias_text"], alias["concept_id"]) for alias in aliases]

        try:
            self.cursor.executemany(
                "INSERT OR IGNORE INTO aliases (alias, concept_id) VALUES (?, ?)",
                records
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error during batch insert into aliases: {e}")
            self.conn.rollback()

    def get_statistics(self) -> Dict[str, Any]:
        """Get alias count statistics."""
        self.cursor.execute("SELECT COUNT(*) FROM aliases")
        count = self.cursor.fetchone()[0]
        return {"total_aliases": count}

    def get_all_aliases(self) -> List[Dict[str, Any]]:
        """Retrieve all aliases for caching."""
        self.cursor.execute("SELECT alias, concept_id FROM aliases")
        return [{"alias_text": row[0], "concept_id": row[1]} for row in self.cursor.fetchall()]

    def close(self):
        """Close connection (managed by KGManager)."""
        pass


if __name__ == "__main__":
    from concept_store import ConceptStore

    print("AliasStore Demo")

    conn = sqlite3.connect(":memory:")
    concept_store = ConceptStore(conn)
    alias_store = AliasStore(conn)

    concept_store.add_concepts_batch([{
        "id": "total_revenue",
        "display_name": "Total Revenue",
        "definition": "Total income from sales"
    }])

    aliases = [
        {"alias_text": "Revenue", "concept_id": "total_revenue"},
        {"alias_text": "Sales", "concept_id": "total_revenue"},
    ]
    alias_store.add_aliases_batch(aliases)
    print(f"Added {len(aliases)} aliases")

    all_aliases = alias_store.get_all_aliases()
    for a in all_aliases:
        print(f"  - '{a['alias_text']}' -> {a['concept_id']}")

    stats = alias_store.get_statistics()
    print(f"Stats: {stats}")

    conn.close()
    print("Demo complete!")
