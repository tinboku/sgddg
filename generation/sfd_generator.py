"""SFD Generator - produces search-optimized structured metadata with KG keyword expansion."""
import json
import re
from typing import Dict, Any, List, Optional

from .llm_adapter import LLMAdapter


class SFDGenerator:
    """Generates Search-Facing Descriptions (SFD) with KG-grounded keyword expansion."""
    def __init__(self, api_key: Optional[str] = None, kg_manager=None):
        self.llm_adapter = LLMAdapter(api_key=api_key)
        self.kg_manager = kg_manager

    def _expand_keywords_from_kg(
        self, columns_data: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Extract synonym and related keyword lists from KG for matched concepts."""
        if not self.kg_manager:
            return {"synonyms": [], "related": []}

        kg_synonyms = set()
        kg_related = set()

        for col in columns_data:
            match_info = col.get("match_info", {})
            if match_info.get("status") != "matched":
                continue

            concept_id = match_info.get("matched_concept_id") or match_info.get("concept", {}).get("id")
            if not concept_id:
                continue

            # 1. Get all aliases for this concept → synonyms
            try:
                alias_store = self.kg_manager.alias_store
                alias_store.cursor.execute(
                    "SELECT alias FROM aliases WHERE concept_id = ?", (concept_id,)
                )
                aliases = [row[0] for row in alias_store.cursor.fetchall()]
                kg_synonyms.update(aliases)
            except Exception:
                pass

            # 2. Get parent concepts (hypernyms) → related
            try:
                parents = self.kg_manager.relationship_store.find_parents(concept_id)
                for parent_id in parents:
                    parent_concept = self.kg_manager.concept_store.get_concept_by_id(parent_id)
                    if parent_concept:
                        kg_related.add(parent_concept.get("display_name", parent_id))
                        # Second-order: get aliases of parent
                        try:
                            alias_store.cursor.execute(
                                "SELECT alias FROM aliases WHERE concept_id = ?", (parent_id,)
                            )
                            parent_aliases = [row[0] for row in alias_store.cursor.fetchall()]
                            kg_related.update(parent_aliases)
                        except Exception:
                            pass
            except Exception:
                pass

            # 3. Get child concepts (hyponyms) → related
            try:
                children = self.kg_manager.relationship_store.find_children(concept_id)
                for child_id in children:
                    child_concept = self.kg_manager.concept_store.get_concept_by_id(child_id)
                    if child_concept:
                        kg_related.add(child_concept.get("display_name", child_id))
            except Exception:
                pass

            # 4. Get related_to relationships → related
            try:
                rel_store = self.kg_manager.relationship_store
                rel_store.cursor.execute(
                    "SELECT target_concept_id FROM relationships WHERE source_concept_id = ? AND relationship_type = 'related_to'",
                    (concept_id,)
                )
                for row in rel_store.cursor.fetchall():
                    related_concept = self.kg_manager.concept_store.get_concept_by_id(row[0])
                    if related_concept:
                        kg_related.add(related_concept.get("display_name", row[0]))
            except Exception:
                pass

        return {
            "synonyms": sorted(list(kg_synonyms)),
            "related": sorted(list(kg_related)),
        }

    def _fix_json_response(self, response_text: str) -> str:
        """Attempts to fix a potentially broken JSON string from the LLM."""
        text = response_text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

        # Find outermost JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return ""
        json_str = match.group(0)

        # Balance braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        json_str += '}' * (open_braces - close_braces)

        # Remove trailing commas before closing brace/bracket
        json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)

        # Validate; if invalid, try replacing single quotes with double quotes
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            json_str = json_str.replace("'", '"')

        return json_str

    def _build_prompt(
        self,
        dataset_name: str,
        columns_data: List[Dict[str, Any]],
        ufd_text: str,
        kg_seed_keywords: Optional[Dict[str, List[str]]] = None,
        dataset_topics: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for structured SFD JSON generation with KG seed keywords."""
        column_contexts = []
        for col in columns_data:
            match_info = col.get("match_info", {})
            sem_profile = col.get("semantic_profile", {})
            kg_concept = match_info.get("concept", {})

            ctx = {
                "column_name": col.get("column_name"),
                "kg_match": {
                    "concept_name": kg_concept.get("display_name"),
                    "score": match_info.get("score"),
                    "id": kg_concept.get("id"),
                    "definition": kg_concept.get("definition"),
                    "aliases": kg_concept.get("aliases", [])
                },
                "semantic_inferred": {
                    "entity_type": sem_profile.get("Identity", {}).get("EntityType"),
                    "domain": sem_profile.get("Identity", {}).get("Domain"),
                    "functional_role": sem_profile.get("Usage", {}).get("FunctionalRole")
                }
            }
            column_contexts.append(ctx)

        # Build KG seed keywords section
        kg_seed_section = ""
        if kg_seed_keywords and (kg_seed_keywords.get("synonyms") or kg_seed_keywords.get("related")):
            kg_seed_section = "\n**Seed Keywords from Knowledge Graph (use these as a starting point, then expand further):**\n"
            if kg_seed_keywords.get("synonyms"):
                kg_seed_section += f"- Synonyms/Aliases: {', '.join(kg_seed_keywords['synonyms'][:30])}\n"
            if kg_seed_keywords.get("related"):
                kg_seed_section += f"- Related/Hypernyms: {', '.join(kg_seed_keywords['related'][:30])}\n"
            kg_seed_section += "\n"

        # Build topic context section
        topic_section = ""
        if dataset_topics:
            topic_section = f"\n**Detected Dataset Topics:** {', '.join(dataset_topics)}\n"

        prompt = f"""
You are an expert Data Cataloger. Your task is to generate a **Search-Facing Description (SFD)** for a dataset.
This SFD is used by search engines and data discovery platforms (like AutoDDG) to index the dataset.
It requires high-recall keywords, domain classification, and precise concept mapping.

**Dataset Context:**
- Name: {dataset_name}
- Human Description: "{ufd_text}"
{topic_section}
**Column Metadata (Physical & Semantic):**
```json
{json.dumps(column_contexts, indent=2)}
```
{kg_seed_section}
**Task Requirements:**
Generate a JSON object containing:
1.  **summary**: A concise, technical summary optimized for keyword matching.
2.  **domain_tags**: A list of high-level domains (e.g., "Finance", "Healthcare", "ESG").
3.  **keywords**: A structured dictionary with:
    -   `core`: Exact terms from the dataset and matched concepts.
    -   `synonyms`: Alternate names, aliases from KG seed keywords above, and additional synonyms you know. INCLUDE ALL seed synonyms provided above.
    -   `related`: Broader terms, industry jargon, related concepts, and hypernyms from seed keywords above. INCLUDE ALL seed related terms provided above.
4.  **schema_metadata**: A list of objects for each column containing:
    -   `column_name`: Original name.
    -   `concept_name`: The canonical business term (from KG match or inference).
    -   `concept_id`: The KG ID if available (or null).
    -   `confidence`: A score (0.0-1.0) reflecting confidence in the concept mapping (use KG score if available).
    -   `description`: A short, search-optimized definition of the column.
    -   `semantic_type`: The inferred type (e.g., "Metric", "Entity", "Temporal").

**Output Format:**
Return ONLY a valid JSON object. No markdown formatting.

{{
    "summary": "...",
    "domain_tags": ["..."],
    "keywords": {{
        "core": ["..."],
        "synonyms": ["..."],
        "related": ["..."]
    }},
    "schema_metadata": [
        {{
            "column_name": "...",
            "concept_name": "...",
            "concept_id": "...",
            "confidence": 0.95,
            "description": "...",
            "semantic_type": "..."
        }},
        ...
    ]
}}
"""
        return prompt.strip()

    def generate(
        self,
        dataset_name: str,
        columns_data: List[Dict[str, Any]],
        ufd: Dict[str, Any],
        dataset_topics: Optional[List[str]] = None,
        relationship_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured SFD with KG-grounded keyword expansion."""
        ufd_text = ufd.get("core_description", {}).get("text", "")

        # Expand keywords from KG before building prompt
        kg_seed_keywords = self._expand_keywords_from_kg(columns_data)
        if kg_seed_keywords.get("synonyms") or kg_seed_keywords.get("related"):
            print(f"   -> KG Keyword Expansion: {len(kg_seed_keywords.get('synonyms', []))} synonyms, "
                  f"{len(kg_seed_keywords.get('related', []))} related terms")

        prompt = self._build_prompt(
            dataset_name, columns_data, ufd_text,
            kg_seed_keywords=kg_seed_keywords,
            dataset_topics=dataset_topics,
        )

        print("   -> Generating Structured Search-Facing Description (SFD)...")
        response_text = self.llm_adapter.generate_description({"prompt": prompt})

        if not response_text:
            return {"text": "Generation Failed", "format": "error"}

        fixed_json = self._fix_json_response(response_text)
        try:
            sfd_json = json.loads(fixed_json)

            # Add schema structure information if relationship_info is provided
            if relationship_info:
                sfd_json["schema_structure"] = self._encode_schema_structure(
                    columns_data, relationship_info
                )
                print(f"   -> Schema structure encoded: {len(relationship_info.get('hierarchies', []))} hierarchies, "
                      f"{len(relationship_info.get('foreign_keys', []))} FK candidates")

            # Create a markdown representation for display purposes
            markdown_text = self._json_to_markdown(sfd_json)
            sfd_json["text"] = markdown_text
            print("   -> SFD generation complete.")
            return sfd_json
        except json.JSONDecodeError:
            print(f"   Warning: Failed to parse SFD JSON. Response: {response_text[:100]}...")
            return {"text": response_text, "format": "raw_text"}

    def generate_kg_enhanced_search_text(
        self,
        dataset_name: str,
        ufd_text: str,
        sfd_json: Dict[str, Any],
        columns_data: List[Dict[str, Any]],
    ) -> str:
        """Generate a KG-enhanced, keyword-rich search-optimized text expansion."""
        # Collect KG context
        kg_concepts_with_defs = []
        kg_expansion = self._expand_keywords_from_kg(columns_data)

        for col in columns_data:
            match_info = col.get("match_info", {})
            if match_info.get("status") != "matched":
                continue
            concept = match_info.get("concept", {})
            name = concept.get("display_name", "")
            defn = concept.get("definition", "")
            if name:
                entry = f"- {name}"
                if defn:
                    entry += f": {defn[:150]}"
                kg_concepts_with_defs.append(entry)

        kg_concepts_text = "\n".join(kg_concepts_with_defs) if kg_concepts_with_defs else "None available"

        all_synonyms = ", ".join(kg_expansion.get("synonyms", [])[:40])
        all_related = ", ".join(kg_expansion.get("related", [])[:40])

        # Existing SFD keywords for seed
        sfd_core = ", ".join(sfd_json.get("keywords", {}).get("core", []))
        sfd_related = ", ".join(sfd_json.get("keywords", {}).get("related", []))

        prompt = f"""You are given a dataset about the topic '{dataset_name}', with the initial description:
{ufd_text[:500]}

The following knowledge graph concepts were matched to the dataset columns:
{kg_concepts_text}

Known synonyms from KG: {all_synonyms}
Related terms from KG: {all_related}
Existing core keywords: {sfd_core}
Existing related keywords: {sfd_related}

Write a comprehensive search-optimized paragraph (300-500 words) that:
1. Includes the exact dataset topic and all related concepts from the knowledge graph
2. Adds synonyms, related terms, and domain-specific vocabulary
3. Includes any variations of key terms that improve searchability
4. Uses the matched KG concepts and their relationships to add domain-specific context
5. Includes acronyms, alternative phrasings, and industry jargon
6. Mentions application scenarios and use cases

Focus less on readability and more on including ALL relevant terms.
Output ONLY the paragraph text, no headers or formatting."""

        try:
            search_text = self.llm_adapter.generate_description({"prompt": prompt})
            if search_text:
                return search_text.strip()
        except Exception as e:
            print(f"   Warning: KG-enhanced search text generation failed: {e}")

        # Fallback: concatenate available terms
        parts = [ufd_text]
        if all_synonyms:
            parts.append(all_synonyms)
        if all_related:
            parts.append(all_related)
        return " ".join(parts)

    def _json_to_markdown(self, sfd_json: Dict[str, Any]) -> str:
        """Converts the rich SFD JSON to a readable Markdown format for preview."""
        md = f"### Summary\n{sfd_json.get('summary', '')}\n\n"

        md += "### Domain Tags\n"
        md += ", ".join([f"`{tag}`" for tag in sfd_json.get('domain_tags', [])]) + "\n\n"

        md += "### Keywords\n"
        kw = sfd_json.get('keywords', {})
        md += f"**Core**: {', '.join(kw.get('core', []))}\n\n"
        md += f"**Synonyms**: {', '.join(kw.get('synonyms', []))}\n\n"
        md += f"**Related**: {', '.join(kw.get('related', []))}\n\n"

        md += "### Schema Metadata\n"
        md += "| Column | Concept | Type | Conf. | Description |\n"
        md += "|--------|---------|------|-------|-------------|\n"

        for col in sfd_json.get('schema_metadata', []):
            md += f"| {col.get('column_name')} | {col.get('concept_name')} | {col.get('semantic_type')} | {col.get('confidence')} | {col.get('description')} |\n"

        return md

    def _encode_schema_structure(
        self,
        columns_data: List[Dict[str, Any]],
        relationship_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Encode schema structure (hierarchies, FKs, dependencies, constraints) into metadata."""
        structure = {
            "hierarchies": [],
            "foreign_keys": [],
            "dependencies": [],
            "constraints": {}
        }

        # Encode hierarchies
        for hierarchy in relationship_info.get('hierarchies', []):
            structure['hierarchies'].append({
                "type": hierarchy.get('type'),
                "path": hierarchy.get('columns'),
                "confidence": hierarchy.get('confidence'),
                "description": hierarchy.get('description')
            })

        # Encode foreign keys
        for fk in relationship_info.get('foreign_keys', []):
            structure['foreign_keys'].append({
                "column": fk.get('column_name'),
                "confidence": fk.get('confidence'),
                "cardinality": fk.get('cardinality_ratio'),
                "null_rate": fk.get('null_rate')
            })

        # Encode dependencies
        for dep in relationship_info.get('dependencies', []):
            structure['dependencies'].append({
                "from": dep.get('from_columns'),
                "to": dep.get('to_column'),
                "type": dep.get('relation_type'),
                "confidence": dep.get('confidence'),
                "verified": dep.get('verified', False)
            })

        # Extract constraints from column profiles
        for col_data in columns_data:
            col_name = col_data.get('column_name')
            physical_profile = col_data.get('physical_profile', {})
            constraints = physical_profile.get('constraints', {})

            if constraints:
                col_constraints = []

                if constraints.get('not_null'):
                    col_constraints.append("NOT NULL")
                if constraints.get('unique'):
                    col_constraints.append("UNIQUE")
                if constraints.get('primary_key_candidate'):
                    col_constraints.append("PRIMARY KEY (candidate)")
                if constraints.get('foreign_key_candidate'):
                    col_constraints.append("FOREIGN KEY (candidate)")

                # Add check constraints
                check = constraints.get('check_constraints', {})
                if check:
                    if check.get('non_negative'):
                        col_constraints.append("CHECK (value >= 0)")
                    if check.get('range'):
                        col_constraints.append(f"CHECK ({check.get('description', 'range constraint')})")

                if col_constraints:
                    structure['constraints'][col_name] = col_constraints

        return structure

if __name__ == "__main__":
    pass
