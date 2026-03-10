from typing import Dict, Any, Optional, List
import json
import re
import sys
import os

from .llm_adapter import LLMAdapter

class SemanticProfiler:
    """LLM-based semantic profiler that synthesizes physical stats and KG matches into structured profiles."""

    TEMPLATE = """
    {
        "Identity": {
            "BestMatchConcept": "The most accurate standard business term (prioritize KG match if valid)",
            "Confidence": "High/Medium/Low based on evidence",
            "EntityType": "High-level entity (e.g., Person, Organization, Location, Event, Product, Metric, Time)",
            "Domain": "Specific domain (e.g., Finance, Healthcare, E-commerce)"
        },
        "Semantics": {
            "Temporal": {
                "isTemporal": boolean,
                "resolution": "e.g., Year, Month, Day, None"
            },
            "Spatial": {
                "isSpatial": boolean,
                "resolution": "e.g., Country, City, Address, None"
            },
            "Unit": "Unit of measurement (e.g., USD, %, kg) or None"
        },
        "Usage": {
            "FunctionalRole": "Dimension (grouping) / Measure (calculation) / Key (identifier) / Attribute (detail)",
            "UsageContext": "How this column is likely used in analysis (e.g., 'Aggregated by sum', 'Used as a filter')"
        },
        "Relation": {
            "Parent": "Name of a potential parent column (hierarchy) or None",
            "Related": "Name of strongly related columns (e.g. Revenue -> Profit) or None"
        },
        "DataQuality": {
            "InferredConstraint": "Business rule inferred from stats (e.g. 'Must be non-negative', 'Unique values')",
            "ContentSummary": "Brief description of the value distribution"
        }
    }
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm_adapter = LLMAdapter(api_key=api_key)

    def _build_prompt(self,
                      column_name: str,
                      sample_values: List[Any],
                      physical_stats: Dict[str, Any],
                      kg_match: Optional[Dict[str, Any]],
                      all_columns: List[str]) -> str:
        """Build prompt for LLM semantic profile generation."""

        # Enhanced stats for LLM consumption
        stats_summary = {
            "data_type": physical_stats.get("data_type"),
            "null_rate": physical_stats.get("null_rate"),
            "unique_count": physical_stats.get("unique_count"),
            "cardinality_ratio": physical_stats.get("cardinality_ratio"),
            "min": physical_stats.get("min"),
            "max": physical_stats.get("max"),
            "distribution_type": physical_stats.get("distribution_type"),
            "temporal_resolution": physical_stats.get("temporal_resolution"),
            "value_pattern": physical_stats.get("value_pattern"),
            "value_range_semantics": physical_stats.get("value_range_semantics"),
        }
        stats_summary = {k: v for k, v in stats_summary.items() if v is not None}
        stats_str = json.dumps(stats_summary, indent=2)

        kg_context = "No direct knowledge graph match found. Infer schema based on data patterns."
        if kg_match and kg_match.get('status') == 'matched':
            concept = kg_match.get('concept', {})
            kg_context = (
                f"**Strong Signal from Knowledge Graph:**\n"
                f"- Suggested Concept: {concept.get('display_name')}\n"
                f"- Definition: {concept.get('definition')}\n"
                f"- Confidence Score: {kg_match.get('score', 0):.2f}\n"
                f"Instruction: Use this concept as the ground truth for 'Identity' if it aligns with the sample values."
            )

        all_cols_str = ", ".join(all_columns) if all_columns else "N/A"

        prompt = f"""
You are an expert dataset semantic analyzer. Your task is to synthesize a rich semantic profile for a specific column.
You must combine physical data statistics with external knowledge graph signals to produce an accurate interpretation.

**Target Column Information:**
- Column Name: `{column_name}`
- Sample Values: `{sample_values}`
- Physical Statistics:
```json
{stats_str}
```

**Dataset Context (for inferring relationships):**
- All Columns: [{all_cols_str}]

**External Knowledge Context:**
{kg_context}

**Analysis Instructions:**
1. **Identity**: Determine what this column represents. If the Knowledge Graph signal is strong and matches the samples, use it.
2. **Semantics**: Check for Temporal (Time) or Spatial (Location) properties.
3. **Usage**: Infer if this is a Metric (for math), a Dimension (for grouping), or a Key (ID).
   - Hint: High cardinality + String often means Key or Dimension. Numeric often means Measure.
4. **Relations**: Look at "All Columns". If you see 'City' and 'Country', link them.
5. **Quality**: Use stats (e.g., min/max, nulls) to infer business rules (e.g. "Positive values only").

**Output Format:**
Your output MUST be a single, valid JSON object that strictly follows this template. Do not include any extra text.

**Template:**
{self.TEMPLATE}

**Your JSON Output:**
"""
        return prompt.strip()

    def _fix_json_response(self, response_text: str) -> str:
        """Attempt to fix a potentially broken JSON string from the LLM."""
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            return ""

        json_str = match.group(0)

        # Balance braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        json_str += '}' * (open_braces - close_braces)

        # Remove trailing commas
        json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)

        return json_str

    def profile_dataset_batch(self,
                              columns_profiles: List[Dict[str, Any]],
                              kg_matches: List[Optional[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Generate semantic profiles for all columns in a single LLM call."""
        all_columns = [p["column_name"] for p in columns_profiles]
        columns_context = []

        for i, profile in enumerate(columns_profiles):
            col_name = profile["column_name"]
            stats = profile.get("statistics", {})
            match_res = kg_matches[i]

            kg_info = "No KG match"
            if match_res and match_res.get('status') == 'matched':
                concept = match_res.get('concept', {})
                kg_info = f"KG Match: {concept.get('display_name')} (Score: {match_res.get('score', 0):.2f})"

            # Compact summary for batch prompt
            enhanced_hints = []
            enhanced_hints.append(f"Nulls: {profile.get('null_rate', 0):.2f}")
            enhanced_hints.append(f"Unique: {profile.get('unique_count', 0)}")
            if profile.get("cardinality_ratio") is not None:
                enhanced_hints.append(f"CardRatio: {profile.get('cardinality_ratio'):.2f}")
            if profile.get("distribution_type"):
                enhanced_hints.append(f"Dist: {profile.get('distribution_type')}")
            if profile.get("temporal_resolution"):
                enhanced_hints.append(f"TemporalRes: {profile.get('temporal_resolution')}")
            if profile.get("value_pattern"):
                enhanced_hints.append(f"Pattern: {profile.get('value_pattern')}")
            if profile.get("value_range_semantics"):
                enhanced_hints.append(f"RangeSem: {profile.get('value_range_semantics')}")

            col_summary = {
                "name": col_name,
                "type": profile.get("data_type"),
                "samples": profile.get("sample_values")[:3],
                "kg_signal": kg_info,
                "stats_hint": ", ".join(enhanced_hints),
            }
            columns_context.append(col_summary)

        context_str = json.dumps(columns_context, indent=2)

        prompt = f"""
You are an expert Data Architect. Analyze the following dataset schema and generate semantic profiles for ALL columns in one go.

**Dataset Columns & Context:**
```json
{context_str}
```

**Instructions:**
1. Analyze each column based on its physical data and Knowledge Graph (KG) signal.
2. If a KG Match is present and valid, use it as the ground truth for "Identity".
3. Infer relationships between columns (e.g. if you see 'City' and 'Country', link them).
4. Output a single JSON object where keys are column names and values are the semantic profiles.

**Semantic Profile Structure (for each column):**
{self.TEMPLATE}

**Output Format:**
{{
    "ColumnName1": {{ ... profile ... }},
    "ColumnName2": {{ ... profile ... }},
    ...
}}
"""
        print(f"    -> Batch synthesizing semantic profiles for {len(all_columns)} columns...")
        raw_response = self.llm_adapter.generate_description({"prompt": prompt})

        if not raw_response:
            print("    -> Warning: Batch LLM call failed.")
            return {}

        fixed_json_str = self._fix_json_response(raw_response)

        try:
            batch_results = json.loads(fixed_json_str)

            def _merge_full_profile(col_name: str, semantic_core: Dict[str, Any]) -> Dict[str, Any]:
                physical = next((p for p in columns_profiles if p.get("column_name") == col_name), {})
                return {
                    "column_name": col_name,
                    "Physical": {
                        "data_type": physical.get("data_type"),
                        "structural_type": physical.get("structural_type"),
                        "total_count": physical.get("total_count"),
                        "null_count": physical.get("null_count"),
                        "null_rate": physical.get("null_rate"),
                        "unique_count": physical.get("unique_count"),
                        "statistics": physical.get("statistics", {}),
                        "sample_values": physical.get("sample_values", []),
                        "unit": physical.get("unit"),
                        "inferred_semantic_type": physical.get("inferred_semantic_type"),
                    },
                    # Core semantic structure from LLM
                    **(semantic_core or {})
                }

            if isinstance(batch_results, dict):
                print(f"    -> Batch generation successful. Received {len(batch_results)} profiles.")
                full_results: Dict[str, Dict[str, Any]] = {}
                for col_name, semantic_core in batch_results.items():
                    full_results[col_name] = _merge_full_profile(col_name, semantic_core)
                return full_results
            elif isinstance(batch_results, list):
                print("    -> Warning: LLM returned a list instead of dict. Attempting to map by order.")
                full_results: Dict[str, Dict[str, Any]] = {}
                for i, semantic_core in enumerate(batch_results):
                    if i < len(all_columns):
                        col_name = all_columns[i]
                        full_results[col_name] = _merge_full_profile(col_name, semantic_core)
                return full_results
            else:
                print(f"    -> Warning: Unexpected JSON structure: {type(batch_results)}")
                return {}
        except json.JSONDecodeError as e:
            print(f"    -> Warning: Failed to parse batch response. Error: {e}")
            return {}

    def profile_column(self,
                       column_profile: Dict[str, Any],
                       all_columns: List[str],
                       kg_match: Optional[Dict[str, Any]] = None,
                       relationship_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate a semantic profile for a single column via LLM."""
        column_name = column_profile["column_name"]
        sample_values = column_profile["sample_values"]
        physical_stats = column_profile.get("statistics", {})

        prompt = self._build_prompt(column_name, sample_values, physical_stats, kg_match, all_columns)

        print(f"    -> Synthesizing semantic profile for '{column_name}'...")

        raw_response = self.llm_adapter.generate_description({"prompt": prompt})

        if not raw_response:
            print(f"    -> Warning: LLM call failed for '{column_name}'.")
            return None

        fixed_json_str = self._fix_json_response(raw_response)

        try:
            semantic_core = json.loads(fixed_json_str)

            # Enhance Relation field with detected relationships if available
            if relationship_info:
                relation_summary = self._extract_relation_from_relationships(
                    column_name, relationship_info
                )
                if 'Relation' in semantic_core:
                    semantic_core['Relation'].update(relation_summary)
                else:
                    semantic_core['Relation'] = relation_summary

            # Merge physical profile with LLM semantic result
            full_profile = {
                "column_name": column_name,
                "Physical": {
                    "data_type": column_profile.get("data_type"),
                    "structural_type": column_profile.get("structural_type"),
                    "total_count": column_profile.get("total_count"),
                    "null_count": column_profile.get("null_count"),
                    "null_rate": column_profile.get("null_rate"),
                    "unique_count": column_profile.get("unique_count"),
                    "statistics": column_profile.get("statistics", {}),
                    "sample_values": column_profile.get("sample_values", []),
                    "unit": column_profile.get("unit"),
                    "inferred_semantic_type": column_profile.get("inferred_semantic_type"),
                    "constraints": column_profile.get("constraints", {}),
                },
                **(semantic_core or {})
            }
            print(f"    -> Semantic profile for '{column_name}' generated successfully.")
            return full_profile
        except json.JSONDecodeError as e:
            print(f"    -> Warning: Failed to parse semantic profile for '{column_name}'. Error: {e}")
            print(f"       Raw response was: {raw_response}")
            print(f"       Fixed JSON was: {fixed_json_str}")
            return None

    def _extract_relation_from_relationships(
        self,
        column_name: str,
        relationship_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relationship information for a specific column from detected relationships."""
        relation = {
            "Parent": None,
            "Children": [],
            "Related": [],
            "Hierarchy": None,
            "Dependencies": {
                "depends_on": [],
                "depended_by": []
            }
        }

        # Extract from hierarchies
        for hierarchy in relationship_info.get('hierarchies', []):
            columns = hierarchy['columns']
            if column_name in columns:
                idx = columns.index(column_name)
                if idx > 0:
                    relation['Parent'] = columns[idx - 1]
                if idx < len(columns) - 1:
                    relation['Children'] = columns[idx + 1:]
                relation['Hierarchy'] = ' -> '.join(columns)
                break

        # Extract from foreign keys
        for fk in relationship_info.get('foreign_keys', []):
            if fk['column_name'] == column_name:
                base_name = column_name.replace('_id', '').replace('_key', '').replace('_code', '')
                all_hierarchies = relationship_info.get('hierarchies', [])
                for h in all_hierarchies:
                    for col in h['columns']:
                        if base_name.lower() in col.lower() and col != column_name:
                            if col not in relation['Related']:
                                relation['Related'].append(col)

        # Extract from dependencies
        for dep in relationship_info.get('dependencies', []):
            if dep['to_column'] == column_name:
                relation['Dependencies']['depends_on'] = dep['from_columns']
            if column_name in dep['from_columns']:
                if dep['to_column'] not in relation['Dependencies']['depended_by']:
                    relation['Dependencies']['depended_by'].append(dep['to_column'])

        # Clean up lists to strings or None
        if not relation['Children']:
            relation['Children'] = None
        else:
            relation['Children'] = ', '.join(relation['Children'])

        if not relation['Related']:
            relation['Related'] = None
        else:
            relation['Related'] = ', '.join(relation['Related'])

        if not relation['Dependencies']['depends_on'] and not relation['Dependencies']['depended_by']:
            del relation['Dependencies']
        else:
            if relation['Dependencies']['depends_on']:
                relation['Dependencies']['depends_on'] = ', '.join(relation['Dependencies']['depends_on'])
            else:
                del relation['Dependencies']['depends_on']
            if relation['Dependencies']['depended_by']:
                relation['Dependencies']['depended_by'] = ', '.join(relation['Dependencies']['depended_by'])
            else:
                del relation['Dependencies']['depended_by']

        return relation
