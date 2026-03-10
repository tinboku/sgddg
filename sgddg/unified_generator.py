"""
Unified metadata generator for SGDDG.

Generates semantic profiles, UFD, and SFD in a single LLM call,
reducing API costs and latency compared to separate calls.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from .prompt_compressor import PromptCompressor, CompressionLevel


class UnifiedMetadataGenerator:
    """Generates semantic profiles + UFD + SFD via a single LLM call."""

    def __init__(self, api_key: Optional[str] = None, use_optimized: bool = False, optimized_config: Optional[Dict] = None):
        self.use_optimized = use_optimized

        # Lazy-import LLM adapter to avoid circular dependencies
        if use_optimized:
            config = optimized_config or {}
            mode = config.get('mode', 'multi')

            try:
                from generation.optimized_llm_adapter import OptimizedLLMAdapter
                self.llm_adapter = OptimizedLLMAdapter(
                    gemini_key=config.get('gemini_key', api_key),
                    openai_key=config.get('openai_key'),
                    use_kg=config.get('use_kg', True),
                    prefer_cheap=config.get('prefer_cheap', True),
                    enable_cache=config.get('enable_cache', True)
                )
            except ImportError:
                logger.warning("Optimized adapter not available, falling back to standard")
                use_optimized = False

        if not use_optimized:
            try:
                from generation.llm_adapter import LLMAdapter
                self.llm_adapter = LLMAdapter(api_key=api_key)
            except ImportError:
                logger.warning("Could not import LLMAdapter")
                self.llm_adapter = None

        self.compressor = PromptCompressor(CompressionLevel.BALANCED)

    def get_api_stats(self) -> Optional[Dict]:
        """Return API usage statistics (optimized adapter only)."""
        if self.use_optimized and hasattr(self.llm_adapter, 'get_stats'):
            return self.llm_adapter.get_stats()
        return None

    def generate_from_metadata(
        self,
        dataset_name: str,
        column_names: List[str],
        column_types: Optional[List[str]] = None,
        kg_matches: Optional[List[Optional[Dict[str, Any]]]] = None,
        dataset_context: str = "",
    ) -> Dict[str, Any]:
        """Simplified entry point for benchmark evaluation.

        Accepts only column names and optional inferred types (no CSV data needed).
        Constructs minimal physical profiles and delegates to generate_all_metadata().
        """
        if column_types is None:
            column_types = [self._infer_type_from_name(name) for name in column_names]

        # Build minimal physical profiles
        physical_profiles = []
        for i, col_name in enumerate(column_names):
            physical_profiles.append({
                "column_name": col_name,
                "data_type": column_types[i] if i < len(column_types) else "string",
                "sample_values": [],
                "null_rate": 0.0,
                "unique_count": 0,
                "total_count": 0,
            })

        if kg_matches is None:
            kg_matches = [{"status": "no_match"} for _ in column_names]
        else:
            while len(kg_matches) < len(column_names):
                kg_matches.append({"status": "no_match"})

        return self.generate_all_metadata(
            dataset_name=dataset_name,
            physical_profiles=physical_profiles,
            kg_matches=kg_matches,
            dataset_context=dataset_context,
        )

    @staticmethod
    def _infer_type_from_name(column_name: str) -> str:
        """Heuristic: infer likely data type from a column name string."""
        name_lower = column_name.lower().replace("_", " ").replace("-", " ")
        numeric_hints = [
            "amount", "price", "cost", "revenue", "profit", "salary", "wage",
            "count", "number", "num", "total", "sum", "avg", "mean", "rate",
            "ratio", "percent", "score", "index", "quantity", "qty", "volume",
            "weight", "height", "width", "length", "area", "size", "age",
            "latitude", "longitude", "lat", "lng", "lon",
        ]
        date_hints = [
            "date", "time", "year", "month", "day", "timestamp", "datetime",
            "created", "updated", "modified", "born", "start", "end",
        ]
        id_hints = ["id", "code", "key", "identifier", "uuid"]

        for hint in numeric_hints:
            if hint in name_lower.split():
                return "float64"
        for hint in date_hints:
            if hint in name_lower.split():
                return "datetime"
        for hint in id_hints:
            if hint in name_lower.split():
                return "string"
        return "string"

    def generate_all_metadata(
        self,
        dataset_name: str,
        physical_profiles: List[Dict[str, Any]],
        kg_matches: List[Optional[Dict[str, Any]]],
        dataset_context: str = ""
    ) -> Dict[str, Any]:
        """Generate all metadata (semantic profiles, UFD, SFD) in a single LLM call."""
        if not self.llm_adapter:
            logger.error("LLM Adapter not initialized")
            return self._get_empty_result()

        prompt = self._build_unified_prompt(
            dataset_name,
            physical_profiles,
            kg_matches,
            dataset_context=dataset_context
        )

        logger.info("Calling LLM for unified metadata generation")

        raw_response = self.llm_adapter.generate_description({"prompt": prompt})

        if not raw_response:
            logger.error("LLM call failed")
            return self._get_empty_result()

        return self._parse_unified_response(raw_response, len(physical_profiles))

    def _build_unified_prompt(
        self,
        dataset_name: str,
        physical_profiles: List[Dict[str, Any]],
        kg_matches: List[Optional[Dict[str, Any]]],
        dataset_context: str = ""
    ) -> str:
        """Build the full prompt that produces semantic profiles, UFD, and SFD."""

        # Compress column data for the prompt
        columns_data = []
        for i, profile in enumerate(physical_profiles):
            columns_data.append({
                "column_name": profile.get("column_name"),
                "profile": profile,
                "kg_match": kg_matches[i] if i < len(kg_matches) else {"status": "no_match"}
            })

        columns_json = self.compressor.format_batch_for_prompt(columns_data)

        prompt = f"""You are an expert data architect and technical writer. Your task is to generate a COMPLETE metadata package for this dataset in a SINGLE JSON response.

**Dataset Name:** {dataset_name}
**Relational Context (Inferred from KG):** {dataset_context if dataset_context else 'N/A'}

**Columns Information:**
```json
{columns_json}
```

---

**Your Task:** Produce ONE JSON object with THREE main sections:

1. **semantic_profiles**: Deep semantic analysis for EACH column
2. **ufd** (User-Facing Description): Natural language description for human readers
3. **sfd** (Search-Facing Description): Structured metadata optimized for search engines

---

**Output Format (MUST be valid JSON):**

```json
{{
  "semantic_profiles": {{
    "ColumnName1": {{
      "Identity": {{
        "BestMatchConcept": "If KG match exists and aligns with samples, use it. Otherwise, infer from samples.",
        "Confidence": "High/Medium/Low (based on evidence quality)",
        "EntityType": "Person/Organization/Location/Event/Product/Metric/Time/Other",
        "Domain": "e.g., Finance, Healthcare, E-commerce, Energy"
      }},
      "Semantics": {{
        "Temporal": {{
          "isTemporal": true/false,
          "resolution": "Year/Month/Day/Hour/None"
        }},
        "Spatial": {{
          "isSpatial": true/false,
          "resolution": "Country/State/City/Address/Coordinates/None"
        }},
        "Unit": "e.g., USD, EUR, kg, meters, percentage, bps, or None"
      }},
      "Usage": {{
        "FunctionalRole": "Dimension (for grouping) / Measure (for calculation) / Key (identifier) / Attribute (descriptive detail)",
        "UsageContext": "How analysts typically use this column (e.g., 'Group by region', 'Sum for total revenue')"
      }},
      "Relation": {{
        "Parent": "Name of hierarchical parent column (e.g., 'City' → parent: 'Country') or null",
        "Related": "Names of semantically related columns (e.g., 'Revenue' ↔ 'Profit') or null"
      }},
      "DataQuality": {{
        "InferredConstraint": "Business rule inferred from statistics (e.g., 'Must be positive', 'Unique identifier')",
        "ContentSummary": "Brief description of value distribution (e.g., 'Ranges from 0 to 1M', 'Mostly categorical with 5 unique values')"
      }}
    }},
    "ColumnName2": {{ ... }},
    ...
  }},

  "ufd": {{
    "title": "Engaging, descriptive title for the dataset (e.g., 'Global Sales Performance Dataset 2023')",
    "core_description": {{
      "text": "Write 2-3 well-structured paragraphs (150-300 words total) that:
      \\n- Explain what this dataset contains and its purpose
      \\n- Highlight key metrics and dimensions
      \\n- Describe typical use cases or analysis scenarios
      \\n- Make it engaging and readable for data consumers (avoid jargon where possible)
      \\n\\nTone: Professional yet accessible, like a blog post introduction."
    }}
  }},

  "sfd": {{
    "summary": "One technical sentence summarizing the dataset (e.g., 'Multi-dimensional sales data with 30 columns covering revenue, customer demographics, and product categories')",
    "domain_tags": ["Primary Domain", "Secondary Domain"],
    "keywords": {{
      "core": ["Most important business concepts (5-10 keywords)"],
      "related": ["Supporting or related concepts (5-10 keywords)"]
    }},
    "search_text": "A comprehensive paragraph (200-500 words) that includes ALL relevant terms, synonyms, related concepts, domain-specific vocabulary, alternative phrasings, and variations of key terms that users might search for. Focus less on readability and more on maximizing keyword coverage. Include: dataset topic and subtopics, column concept names and their synonyms, domain jargon, acronyms, related industry terms, application scenarios, and any alternative ways someone might describe this dataset. Optimize for BM25 keyword search retrieval.",
    "schema_metadata": [
      {{
        "column_name": "ColumnName1",
        "concept_name": "Standardized concept name (from KG or inferred)",
        "confidence": 0.9,
        "semantic_type": "Dimension/Measure/Key/Attribute",
        "description": "One clear sentence describing the column"
      }},
      {{ ... }}
    ]
  }}
}}
```

---

**Critical Instructions:**

1. **Trust KG Matches (Tiered):**
   - **EXACT/RELATED strength matches** (high confidence): ALWAYS use them for "BestMatchConcept". Include their aliases in SFD keywords.
   - **CONTEXTUAL strength matches** (medium confidence): Use them if they align with column context, but verify against any sample values.
   - **UNCERTAIN or low-score matches**: Treat as hints only. Prefer your own inference over uncertain KG matches.
   - **Filtered/Skipped matches**: Ignore these — they were excluded by the routing pipeline.

2. **Infer Relationships:** Look across ALL columns to find hierarchies (e.g., City → Country) and correlations (e.g., Revenue ↔ Profit).

3. **UFD Quality:**
   - Must be 150-300 words
   - Use natural, flowing language
   - Avoid bullet points
   - Make it interesting to read

4. **SFD Precision:**
   - Keywords should be highly relevant
   - Schema metadata must cover ALL columns
   - Optimize for search engine indexing
   - **search_text is CRITICAL**: It must be a dense paragraph stuffed with every relevant term, synonym, related concept, and domain vocabulary. Think of it as a search-optimized expansion of the dataset description. Include concepts and synonyms from the KG matches above.

5. **JSON Strictness:**
   - Output ONLY the JSON object
   - No extra text before or after
   - Ensure all quotes are properly escaped
   - Validate syntax

---

**Your JSON Output (start with {{ and end with }}):**
"""

        return prompt.strip()

    def _parse_unified_response(
        self,
        raw_response: str,
        expected_columns: int
    ) -> Dict[str, Any]:
        """Parse the unified LLM response into structured metadata."""
        json_str = self._extract_json(raw_response)

        if not json_str:
            logger.error("Failed to extract JSON from LLM response")
            return self._get_empty_result()

        try:
            result = json.loads(json_str)
            validation_ok = self._validate_result(result, expected_columns)

            if validation_ok:
                self._print_result_summary(result)
                return result
            else:
                logger.warning("Validation warnings in parsed result")
                return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._get_empty_result()

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response, handling markdown code blocks and raw JSON."""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*\})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)

        return None

    def _validate_result(
        self,
        result: Dict[str, Any],
        expected_columns: int
    ) -> bool:
        """Validate completeness of the generated metadata."""
        warnings = []

        required_keys = ["semantic_profiles", "ufd", "sfd"]
        for key in required_keys:
            if key not in result:
                warnings.append(f"Missing top-level key: {key}")

        profiles = result.get("semantic_profiles", {})
        if not isinstance(profiles, dict):
            warnings.append("semantic_profiles is not a dict")
        elif len(profiles) != expected_columns:
            warnings.append(
                f"semantic_profiles count mismatch: expected {expected_columns}, got {len(profiles)}"
            )

        ufd = result.get("ufd", {})
        if not ufd.get("title"):
            warnings.append("UFD missing title")
        if not ufd.get("core_description", {}).get("text"):
            warnings.append("UFD missing core_description.text")

        sfd = result.get("sfd", {})
        if not sfd.get("summary"):
            warnings.append("SFD missing summary")
        if not sfd.get("schema_metadata"):
            warnings.append("SFD missing schema_metadata")

        if warnings:
            for w in warnings:
                logger.warning(f"Validation: {w}")
            return False

        return True

    def _print_result_summary(self, result: Dict[str, Any]):
        """Log a brief summary of the generated metadata."""
        profiles = result.get("semantic_profiles", {})
        ufd = result.get("ufd", {})
        sfd = result.get("sfd", {})
        ufd_text = ufd.get("core_description", {}).get("text", "")

        logger.info(
            f"Generated: {len(profiles)} profiles, "
            f"UFD {len(ufd_text.split())} words, "
            f"SFD {len(sfd.get('schema_metadata', []))} schema entries, "
            f"{len(sfd.get('keywords', {}).get('core', []))} keywords"
        )

    def _get_empty_result(self) -> Dict[str, Any]:
        """Return an empty fallback result."""
        return {
            "semantic_profiles": {},
            "ufd": {
                "title": "Untitled Dataset",
                "core_description": {"text": "Description unavailable"}
            },
            "sfd": {
                "summary": "No summary available",
                "domain_tags": [],
                "keywords": {"core": [], "related": []},
                "schema_metadata": []
            }
        }


