"""Topic Detector - detects dataset-level topics and primary domain from column schema."""

import json
import re
from typing import Dict, List, Any, Optional

try:
    from generation.llm_adapter import LLMAdapter
except ImportError:
    LLMAdapter = None


# Keyword-based domain detection as a fast fallback
DOMAIN_KEYWORDS = {
    "finance": [
        "revenue", "profit", "income", "expense", "cost", "price", "sales",
        "stock", "share", "dividend", "ebitda", "margin", "asset", "liability",
        "equity", "debt", "tax", "interest", "capital", "investment", "portfolio",
        "return", "yield", "market_cap", "pe_ratio", "eps", "roe", "roa",
        "cash_flow", "balance_sheet", "fiscal", "quarter", "annual_report",
    ],
    "healthcare": [
        "patient", "diagnosis", "treatment", "symptom", "disease", "drug",
        "medication", "hospital", "clinic", "doctor", "nurse", "medical",
        "health", "clinical", "bmi", "blood_pressure", "heart_rate", "dosage",
        "prescription", "surgery", "lab", "test_result", "mortality",
    ],
    "esg": [
        "emission", "carbon", "co2", "scope_1", "scope_2", "scope_3",
        "sustainability", "renewable", "waste", "water", "energy",
        "diversity", "governance", "esg", "ghg", "climate", "environmental",
        "social", "turnover_rate", "safety", "pollution",
    ],
    "ecommerce": [
        "order", "cart", "product", "customer", "purchase", "checkout",
        "payment", "shipping", "delivery", "item", "sku", "catalog",
        "rating", "review", "wishlist", "discount", "coupon", "refund",
    ],
    "education": [
        "student", "course", "grade", "score", "enrollment", "teacher",
        "school", "university", "gpa", "credit", "semester", "exam",
        "assignment", "lecture", "degree", "major", "curriculum",
    ],
    "real_estate": [
        "property", "house", "apartment", "rent", "mortgage", "sqft",
        "bedroom", "bathroom", "listing", "price", "zip_code", "neighborhood",
    ],
    "transportation": [
        "flight", "airline", "route", "passenger", "vehicle", "trip",
        "distance", "speed", "fuel", "departure", "arrival", "delay",
    ],
}


class TopicDetector:
    """Detects dataset-level topics and primary domain using keyword heuristics or LLM."""

    def __init__(self, api_key: Optional[str] = None):
        self.llm_adapter = None
        if LLMAdapter and api_key:
            try:
                self.llm_adapter = LLMAdapter(api_key=api_key)
            except Exception:
                pass

    def _detect_domain_by_keywords(
        self, column_names: List[str], sample_values: Optional[List[List[Any]]] = None
    ) -> Dict[str, float]:
        """
        Fast keyword-based domain detection.
        Returns domain -> confidence score mapping.
        """
        # Normalize column names
        normalized = []
        for name in column_names:
            tokens = name.lower().replace("-", "_").split("_")
            normalized.extend(tokens)
            normalized.append(name.lower())

        domain_scores: Dict[str, float] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for token in normalized if token in keywords)
            if matches > 0:
                domain_scores[domain] = matches / len(column_names)

        return domain_scores

    def detect_topics_fast(
        self,
        column_names: List[str],
        sample_values: Optional[List[List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Fast topic detection using keyword heuristics only (no LLM call).

        Returns:
            Dict with 'topics' (list of topic phrases), 'primary_domain' (str),
            and 'domain_scores' (dict).
        """
        domain_scores = self._detect_domain_by_keywords(column_names, sample_values)

        if not domain_scores:
            return {
                "topics": [" ".join(column_names[:3])],
                "primary_domain": "general",
                "domain_scores": {},
            }

        # Sort domains by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]

        # Generate topic phrases from top domains
        topics = []
        for domain, score in sorted_domains[:3]:
            if score > 0.05:
                topics.append(domain.replace("_", " ").title() + " Data")

        return {
            "topics": topics or ["General Dataset"],
            "primary_domain": primary_domain,
            "domain_scores": dict(sorted_domains),
        }

    def detect_topics_llm(
        self,
        dataset_name: str,
        column_names: List[str],
        sample_values: Optional[List[List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        LLM-powered topic detection for richer, more accurate results.
        Falls back to keyword-based detection if LLM is unavailable.
        """
        if not self.llm_adapter:
            return self.detect_topics_fast(column_names, sample_values)

        # Build concise context
        columns_info = []
        for i, name in enumerate(column_names):
            info = {"name": name}
            if sample_values and i < len(sample_values):
                info["samples"] = sample_values[i][:3]
            columns_info.append(info)

        prompt = f"""Analyze this dataset schema and identify its topics and domain.

Dataset: {dataset_name}
Columns: {json.dumps(columns_info, indent=1)}

Return a JSON object with:
1. "topics": A list of 2-3 specific topic phrases (e.g., "Corporate Financial Performance", "ESG Carbon Emissions Tracking")
2. "primary_domain": One of: finance, healthcare, esg, ecommerce, education, real_estate, transportation, technology, government, general

Return ONLY valid JSON, no extra text.
{{"topics": ["..."], "primary_domain": "..."}}"""

        try:
            response = self.llm_adapter.generate_description({"prompt": prompt})
            if not response:
                return self.detect_topics_fast(column_names, sample_values)

            # Try to parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                # Validate structure
                if "topics" in result and "primary_domain" in result:
                    return {
                        "topics": result["topics"],
                        "primary_domain": result["primary_domain"],
                        "domain_scores": {},
                    }
        except Exception:
            pass

        return self.detect_topics_fast(column_names, sample_values)

    def detect(
        self,
        dataset_name: str,
        column_names: List[str],
        sample_values: Optional[List[List[Any]]] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entry point for topic detection.

        Args:
            dataset_name: Name of the dataset.
            column_names: List of column names.
            sample_values: Optional list of sample value lists per column.
            use_llm: Whether to use LLM for richer topic detection.

        Returns:
            Dict with 'topics', 'primary_domain', 'domain_scores'.
        """
        if use_llm and self.llm_adapter:
            return self.detect_topics_llm(dataset_name, column_names, sample_values)
        return self.detect_topics_fast(column_names, sample_values)
