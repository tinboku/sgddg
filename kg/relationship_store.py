"""Relationship Store - manages concept-to-concept relationships (is_a, related_to, etc.)."""

import sqlite3
from typing import List, Dict, Any


class RelationshipStore:
    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                source_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                target_concept_id TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                evidence TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_concept_id, relationship_type, target_concept_id)
            )
        """)
        self.conn.commit()

    def add_relationships_batch(self, relationships: List[Dict[str, Any]]):
        if not relationships:
            return
        try:
            records = [
                (
                    rel["source_concept_id"],
                    rel["relationship_type"],
                    rel["target_concept_id"],
                    rel.get("confidence", 0.0),
                    rel.get("evidence")
                )
                for rel in relationships
            ]
            self.cursor.executemany(
                """INSERT OR REPLACE INTO relationships
                   (source_concept_id, relationship_type, target_concept_id, confidence, evidence)
                   VALUES (?, ?, ?, ?, ?)""",
                records
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error during batch insert into relationships: {e}")
            self.conn.rollback()

    def find_parents(self, concept_id: str, relationship_type: str = 'is_a') -> List[str]:
        self.cursor.execute(
            "SELECT target_concept_id FROM relationships WHERE source_concept_id = ? AND relationship_type = ?",
            (concept_id, relationship_type)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def find_children(self, concept_id: str, relationship_type: str = 'is_a') -> List[str]:
        self.cursor.execute(
            "SELECT source_concept_id FROM relationships WHERE target_concept_id = ? AND relationship_type = ?",
            (concept_id, relationship_type)
        )
        return [row[0] for row in self.cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        self.cursor.execute("SELECT COUNT(*) FROM relationships")
        count = self.cursor.fetchone()[0]
        return {"total_relationships": count}
