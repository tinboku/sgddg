"""Knowledge Graph Enhancer - shared KG lookup for LLM adapters."""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List


class KnowledgeGraphEnhancer:
    """Searches the domain KG (SQLite) for concepts related to a query."""

    def __init__(self, kg_db_path: Optional[str] = None):
        if kg_db_path is None:
            project_root = Path(__file__).parent.parent
            kg_db_path = project_root / "data" / "domain_knowledge.db"

        self.kg_db_path = str(kg_db_path)
        self.cache: Dict[str, list] = {}

    def search_concepts(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Keyword search for concepts in the KG."""
        cache_key = f"{query}_{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            conn = sqlite3.connect(self.kg_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, display_name, definition
                FROM concepts
                WHERE display_name LIKE ? OR definition LIKE ?
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", top_k))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'definition': row[2]
                })

            conn.close()
            self.cache[cache_key] = results
            return results

        except Exception as e:
            print(f"WARNING: KG search failed: {e}")
            return []

    def get_related_concepts(self, concept_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve related concepts via the relationships table."""
        try:
            conn = sqlite3.connect(self.kg_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT c.id, c.display_name, c.definition, r.relationship_type
                FROM relationships r
                JOIN concepts c ON r.target_concept_id = c.id
                WHERE r.source_concept_id = ?
                LIMIT ?
            """, (concept_id, top_k))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'definition': row[2],
                    'relation': row[3]
                })

            conn.close()
            return results

        except Exception as e:
            print(f"WARNING: Relationship lookup failed: {e}")
            return []
