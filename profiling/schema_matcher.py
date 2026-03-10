import re
from typing import Dict, Any, List, Optional

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from ..kg.kg_manager import KnowledgeGraphManager

class SchemaMatcher:
    """Cascading column-to-concept matcher: exact, fuzzy, then semantic vector search."""
    def __init__(self, kg_manager: KnowledgeGraphManager):
        self.kg_manager = kg_manager
        print("    -> Caching concepts and aliases for matching...")
        self._concept_cache = self.kg_manager.concept_store.get_all_concepts()
        self._alias_cache = self.kg_manager.alias_store.get_all_aliases()
        
        self._name_to_concept = {}
        for c in self._concept_cache:
            # Try to get a valid name: canonical_name -> display_name -> id
            name = c.get('canonical_name') or c.get('display_name') or c.get('id')
            if name:
                self._name_to_concept[self._normalize_string(name)] = c
        
        self._alias_to_concept_id = {
            self._normalize_string(a['alias_text']): a['concept_id'] for a in self._alias_cache
        }
        self._concept_id_to_concept = {c['id']: c for c in self._concept_cache}
        print(f"    -> Caching complete. Found {len(self._concept_cache)} concepts and {len(self._alias_cache)} aliases.")

    def _normalize_string(self, s: str) -> str:
        """Lowercase, remove punctuation, and extra whitespace."""
        s = str(s).lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def match_column(self, column_profile: Dict[str, Any], semantic_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Attempts to find a concept match for a column using a cascading strategy.
        """
        column_name = column_profile["column_name"]
        normalized_column_name = self._normalize_string(column_name)
        
        print(f"\n  - Matching column: '{column_name}'")

        # Stage 1: Exact Match
        match_result = self._exact_match(normalized_column_name)
        if match_result:
            print(f"    -> SUCCESS (Stage 1): Found exact match '{match_result['concept']['display_name']}'")
            return match_result

        # Stage 2: Fuzzy Match
        match_result = self._fuzzy_match(normalized_column_name)
        if match_result:
            print(f"    -> SUCCESS (Stage 2): Found fuzzy match '{match_result['concept']['display_name']}' with score {match_result['score']}")
            return match_result
            
        # Stage 3: Semantic Match
        match_result = self._semantic_match(column_name, semantic_threshold)
        if match_result:
            print(f"    -> SUCCESS (Stage 3): Found semantic match '{match_result['concept']['display_name']}' with score {match_result['score']:.2f}")
            return match_result

        return {"status": "no_match", "reason": "No match found in any stage."}

    def _exact_match(self, normalized_name: str) -> Optional[Dict[str, Any]]:
        """Matches against canonical names and aliases."""
        if normalized_name in self._name_to_concept:
            concept = self._name_to_concept[normalized_name]
            return {"status": "matched", "method": "exact_name", "score": 1.0, "concept": concept}
        
        if normalized_name in self._alias_to_concept_id:
            concept_id = self._alias_to_concept_id[normalized_name]
            concept = self._concept_id_to_concept[concept_id]
            return {"status": "matched", "method": "exact_alias", "score": 1.0, "concept": concept}
            
        return None

    def _fuzzy_match(self, normalized_name: str, score_cutoff: int = 95) -> Optional[Dict[str, Any]]:
        """Fuzzy matches against canonical names and aliases.
        Uses fuzz.ratio (strict character-level) instead of WRatio to reduce false positives.
        Includes length ratio check to avoid matching short names to long concepts."""
        if not RAPIDFUZZ_AVAILABLE:
            return None

        def _length_ratio_ok(a: str, b: str, min_ratio: float = 0.5) -> bool:
            """Reject matches where lengths differ drastically."""
            la, lb = len(a), len(b)
            if la == 0 or lb == 0:
                return False
            return min(la, lb) / max(la, lb) >= min_ratio

        # Fuzzy match against canonical names
        best_name_match = process.extractOne(normalized_name, self._name_to_concept.keys(), scorer=fuzz.ratio, score_cutoff=score_cutoff)
        if best_name_match and _length_ratio_ok(normalized_name, best_name_match[0]):
            concept = self._name_to_concept[best_name_match[0]]
            return {"status": "matched", "method": "fuzzy_name", "score": best_name_match[1] / 100, "concept": concept}

        # Fuzzy match against aliases
        best_alias_match = process.extractOne(normalized_name, self._alias_to_concept_id.keys(), scorer=fuzz.ratio, score_cutoff=score_cutoff)
        if best_alias_match and _length_ratio_ok(normalized_name, best_alias_match[0]):
            concept_id = self._alias_to_concept_id[best_alias_match[0]]
            concept = self._concept_id_to_concept[concept_id]
            return {"status": "matched", "method": "fuzzy_alias", "score": best_alias_match[1] / 100, "concept": concept}

        return None

    def _semantic_match(self, column_name: str, threshold: float) -> Optional[Dict[str, Any]]:
        """Semantic vector search."""
        search_results = self.kg_manager.vector_store.search(column_name, top_k=1)
        if search_results and search_results[0]['score'] >= threshold:
            top_result = search_results[0]
            return {
                "status": "matched",
                "method": "semantic",
                "score": top_result['score'],
                "concept": top_result['metadata']
            }
        elif search_results:
             # Using .get just in case metadata structure varies
             display_name = search_results[0].get('metadata', {}).get('display_name', 'Unknown')
             print(f"    -> INFO: Top semantic candidate '{display_name}' had score {search_results[0]['score']:.2f}, below threshold {threshold}.")

        return None

