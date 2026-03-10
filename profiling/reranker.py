"""Two-stage re-ranking pipeline: local Cross-Encoder followed by listwise LLM re-ranking."""

import os
import json
import re
from typing import Dict, Any, List, Optional

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

try:
    from generation.llm_adapter import LLMAdapter
except ImportError:
    LLMAdapter = None


# --- Listwise LLM Re-ranking Prompt ---
LISTWISE_RERANK_PROMPT = """You are an expert data analyst. Rank the following candidate concepts by how well they match the given dataset column. Return a JSON array of concept IDs ordered from BEST match to WORST match, with confidence scores.

**Dataset Column:**
- Column Name: `{column_name}`
- Data Type: `{data_type}`
- Sample Values: `{sample_values}`

**Candidate Concepts:**
{candidates_text}

**Scoring Guidelines:**
- 1.0: Perfect match (column name and values directly correspond to concept)
- 0.7-0.9: Strong match with minor ambiguity
- 0.4-0.6: Weak or indirect match
- 0.0-0.3: Very unlikely match

**Return ONLY a valid JSON array, no extra text:**
[
  {{"concept_id": "...", "match_score": 0.95, "justification": "..."}},
  {{"concept_id": "...", "match_score": 0.6, "justification": "..."}}
]
"""


class ReRanker:
    """Two-stage re-ranking: Cross-Encoder for fast local scoring, then listwise LLM for final ranking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder: bool = True,
        use_llm_rerank: bool = True,
    ):
        self.cross_encoder = None
        self.llm_adapter = None
        self.use_cross_encoder = use_cross_encoder
        self.use_llm_rerank = use_llm_rerank

        # Stage 1: Cross-Encoder initialization
        if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                print(f"Cross-Encoder loaded: {cross_encoder_model}")
            except Exception as e:
                print(f"Warning: Could not load Cross-Encoder: {e}")

        # Stage 2: LLM adapter for listwise re-ranking
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if use_llm_rerank and resolved_key:
            try:
                if LLMAdapter:
                    self.llm_adapter = LLMAdapter(api_key=resolved_key)
                    print("LLM adapter initialized for listwise re-ranking.")
            except Exception as e:
                print(f"Warning: Could not initialize LLM adapter: {e}")

    def _cross_encoder_score(
        self, column_profile: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Score candidates using local Cross-Encoder.
        Creates query-document pairs and scores them.
        """
        if not self.cross_encoder:
            return candidates

        column_name = column_profile.get("column_name", "")
        samples = str(column_profile.get("sample_values", [])[:5])
        query = f"{column_name} {samples}"

        pairs = []
        for candidate in candidates:
            doc = f"{candidate.get('display_name', '')}. {candidate.get('definition', '')}"
            pairs.append([query, doc])

        try:
            scores = self.cross_encoder.predict(pairs)

            for i, candidate in enumerate(candidates):
                candidate["cross_encoder_score"] = float(scores[i])

            # Sort by cross-encoder score descending
            candidates.sort(key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)

            print(f"    - Cross-Encoder scored {len(candidates)} candidates")

        except Exception as e:
            print(f"    - Warning: Cross-Encoder scoring failed: {e}")

        return candidates

    def _listwise_llm_rerank(
        self, column_profile: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Listwise LLM re-ranking.
        Sends all top candidates in a single prompt for comparative ranking.
        """
        if not self.llm_adapter or not candidates:
            return candidates

        # Build candidates text for prompt
        candidates_lines = []
        for i, c in enumerate(candidates):
            aliases = c.get("aliases", [])
            alias_str = f" (aliases: {', '.join(aliases[:5])})" if aliases else ""
            candidates_lines.append(
                f"{i+1}. ID: `{c.get('id', 'unknown')}` | "
                f"Name: `{c.get('display_name', '')}` | "
                f"Definition: {c.get('definition', 'N/A')}{alias_str}"
            )

        prompt = LISTWISE_RERANK_PROMPT.format(
            column_name=column_profile.get("column_name", ""),
            data_type=column_profile.get("data_type", "unknown"),
            sample_values=str(column_profile.get("sample_values", [])[:5]),
            candidates_text="\n".join(candidates_lines),
        )

        try:
            response = self.llm_adapter.generate_description({"prompt": prompt})
            if not response:
                return candidates

            # Parse JSON array from response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)

            # Find JSON array
            array_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if not array_match:
                return candidates

            rankings = json.loads(array_match.group(0))

            # Apply LLM scores back to candidates
            id_to_result = {r.get("concept_id"): r for r in rankings}
            for candidate in candidates:
                cid = candidate.get("id", "")
                if cid in id_to_result:
                    candidate["rerank_score"] = id_to_result[cid].get("match_score", 0.0)
                    candidate["rerank_justification"] = id_to_result[cid].get("justification", "")

            # Sort by rerank score
            candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            print(f"    - Listwise LLM re-ranked {len(candidates)} candidates")

        except Exception as e:
            print(f"    - Warning: Listwise LLM re-ranking failed: {e}")
            # Fall back to cross-encoder scores if available
            for c in candidates:
                if "cross_encoder_score" in c and "rerank_score" not in c:
                    c["rerank_score"] = c["cross_encoder_score"]

        return candidates

    def rerank_candidate(
        self, column_profile: Dict[str, Any], concept: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy single-candidate re-ranking (pointwise).
        Used as fallback when neither cross-encoder nor LLM is available.
        """
        if self.cross_encoder:
            column_name = column_profile.get("column_name", "")
            samples = str(column_profile.get("sample_values", [])[:5])
            query = f"{column_name} {samples}"
            doc = f"{concept.get('display_name', '')}. {concept.get('definition', '')}"

            try:
                score = float(self.cross_encoder.predict([[query, doc]])[0])
                # Normalize cross-encoder score to 0-1 range (sigmoid-like)
                normalized = 1.0 / (1.0 + 2.718 ** (-score))
                return {
                    "match_score": normalized,
                    "justification": f"Cross-encoder score: {score:.3f} (normalized: {normalized:.3f})",
                }
            except Exception:
                pass

        # Fallback to vector score
        return {
            "match_score": concept.get("score", 0.0),
            "justification": "Fallback to vector similarity score.",
        }

    def find_best_match(
        self,
        column_profile: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        cross_encoder_top_k: int = 10,
        llm_top_k: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Two-stage re-ranking pipeline.
        Stage 1: Cross-Encoder narrows to top-K candidates (fast, local)
        Stage 2: Listwise LLM ranks top-K for final ordering (single API call)
        """
        if not candidates:
            return None

        # Stage 1: Cross-Encoder scoring
        if self.use_cross_encoder and self.cross_encoder:
            candidates = self._cross_encoder_score(column_profile, candidates)
            candidates = candidates[:cross_encoder_top_k]

        # Stage 2: Listwise LLM re-ranking (only top candidates)
        if self.use_llm_rerank and self.llm_adapter:
            top_for_llm = candidates[:llm_top_k]
            top_for_llm = self._listwise_llm_rerank(column_profile, top_for_llm)
            # Merge back
            candidates[:llm_top_k] = top_for_llm

        # If no rerank_score was assigned, fall back to pointwise scoring
        for candidate in candidates:
            if "rerank_score" not in candidate:
                result = self.rerank_candidate(column_profile, candidate)
                candidate["rerank_score"] = result.get("match_score", 0.0)
                candidate["rerank_justification"] = result.get("justification", "")

            vector_score = candidate.get("score", 0.0)
            final_score = candidate.get("rerank_score", 0.0)
            print(
                f"    - Candidate: '{candidate.get('display_name')}' | "
                f"Vector: {vector_score:.2f} | "
                f"Re-ranked: {final_score:.2f}"
            )

        # Sort final results
        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return candidates[0] if candidates else None
