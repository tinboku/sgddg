"""
Context-aware matcher for SGDDG.

When standard vector retrieval fails, infers concept definitions from
neighboring matched columns and domain patterns.
"""

from typing import Dict, Any, List, Optional
import re


class ContextInferenceEngine:
    """Infers temporary concept definitions for unmatched columns using domain context."""

    # Domain keyword and column pattern dictionaries
    DOMAIN_PATTERNS = {
        "Healthcare": {
            "keywords": ["patient", "diagnosis", "treatment", "medical", "hospital", "clinic",
                        "doctor", "nurse", "prescription", "symptom", "disease", "health",
                        "bmi", "blood", "heart", "surgery", "insurance", "premium", "charges"],
            "column_patterns": {
                "charges": "Medical service fees or insurance premium amounts",
                "premium": "Insurance premium payment amount",
                "copay": "Patient's out-of-pocket payment for medical services",
                "deductible": "Amount patient must pay before insurance coverage begins",
                "bmi": "Body Mass Index - health metric calculated from height and weight",
                "smoker": "Tobacco smoking status indicator (yes/no or frequency)",
                "patient_id": "Unique identifier for a patient record"
            }
        },
        "Finance": {
            "keywords": ["revenue", "profit", "price", "stock", "market", "trading", "investment",
                        "portfolio", "dividend", "yield", "return", "risk", "asset", "liability",
                        "open", "high", "low", "close", "volume", "ticker", "symbol"],
            "column_patterns": {
                "charges": "Financial charges or fees applied to an account",
                "open": "Opening price at the start of a trading period",
                "high": "Highest price reached during a trading period",
                "low": "Lowest price reached during a trading period",
                "close": "Closing price at the end of a trading period",
                "volume": "Total number of shares or contracts traded",
                "ticker": "Stock ticker symbol - unique identifier for a traded security",
                "symbol": "Trading symbol for a financial instrument"
            }
        },
        "E-commerce": {
            "keywords": ["product", "order", "customer", "purchase", "cart", "checkout", "payment",
                        "shipping", "delivery", "inventory", "sku", "price", "discount", "coupon"],
            "column_patterns": {
                "charges": "Total charges for a purchase including taxes and fees",
                "customer_id": "Unique identifier for a customer account",
                "order_id": "Unique identifier for a purchase order"
            }
        }
    }

    def infer_domain_from_neighbors(self, matched_columns: List[Dict[str, Any]]) -> Optional[str]:
        """Infer the dataset's primary domain from already-matched columns."""
        if not matched_columns:
            return None

        # Gather text from matched columns
        all_text = []
        for col in matched_columns:
            all_text.append(col.get("column_name", "").lower())
            if col.get("concept"):
                all_text.append(col["concept"].get("display_name", "").lower())
                all_text.append(col["concept"].get("definition", "").lower())

        combined_text = " ".join(all_text)

        # Score each domain by keyword overlap
        domain_scores = {}
        for domain, config in self.DOMAIN_PATTERNS.items():
            score = sum(1 for kw in config["keywords"] if kw in combined_text)
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] >= 2:
                return best_domain

        return None

    def generate_concept_definition(
        self,
        column_name: str,
        sample_values: List[Any],
        data_type: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a temporary concept definition from context and domain patterns."""
        normalized_name = column_name.lower().strip()

        # Try domain-specific patterns first
        if domain and domain in self.DOMAIN_PATTERNS:
            patterns = self.DOMAIN_PATTERNS[domain]["column_patterns"]
            if normalized_name in patterns:
                return {
                    "id": f"inferred_{normalized_name}",
                    "display_name": column_name.title(),
                    "definition": patterns[normalized_name],
                    "domain": domain,
                    "source": "context_inference",
                    "confidence": 0.75
                }

        # Fall back to generic inference
        definition = self._generate_generic_definition(column_name, sample_values, data_type)

        return {
            "id": f"inferred_{normalized_name}",
            "display_name": column_name.title(),
            "definition": definition,
            "domain": domain or "General",
            "source": "context_inference",
            "confidence": 0.60
        }

    def _generate_generic_definition(self, column_name: str, sample_values: List[Any], data_type: str) -> str:
        """Generate a generic definition from column name, samples, and type."""
        value_desc = ""
        if sample_values:
            if data_type in ["integer", "float"]:
                try:
                    numeric_vals = [float(v) for v in sample_values if v is not None]
                    if numeric_vals:
                        min_val = min(numeric_vals)
                        max_val = max(numeric_vals)
                        value_desc = f" (typical range: {min_val:.2f} to {max_val:.2f})"
                except:
                    pass
            elif data_type == "string":
                unique_vals = list(set(str(v) for v in sample_values[:5] if v is not None))
                if len(unique_vals) <= 5:
                    value_desc = f" (possible values: {', '.join(unique_vals)})"

        name_parts = re.split(r'[_\s]+', column_name.lower())
        readable_name = ' '.join(name_parts).title()

        return f"{readable_name} - a {data_type} field{value_desc}"


class ContextAwareMatcher:
    """Wraps a base matcher with context-based fallback inference."""

    def __init__(self, base_matcher):
        """
        Args:
            base_matcher: The underlying SchemaMatcher instance.
        """
        self.base_matcher = base_matcher
        self.inference_engine = ContextInferenceEngine()
        self.matched_columns = []  # Accumulates successful matches for context

    def match_column_with_context(
        self,
        column_profile: Dict[str, Any],
        semantic_threshold: float = 0.7,
        enable_inference: bool = True
    ) -> Dict[str, Any]:
        """Match a column, falling back to context inference on no-match."""
        # Try standard matching first
        result = self.base_matcher.match_column(column_profile, semantic_threshold)

        # Record successful matches for later context inference
        if result.get("status") == "matched":
            self.matched_columns.append({
                "column_name": column_profile["column_name"],
                "concept": result.get("concept"),
                "score": result.get("score")
            })
            return result

        # Fallback: context-based inference
        if enable_inference and result.get("status") == "no_match":
            print(f"    -> FALLBACK: Attempting context-based inference...")

            domain = self.inference_engine.infer_domain_from_neighbors(self.matched_columns)

            if domain:
                print(f"    -> Inferred domain: {domain}")

                inferred_concept = self.inference_engine.generate_concept_definition(
                    column_name=column_profile["column_name"],
                    sample_values=column_profile.get("sample_values", []),
                    data_type=column_profile.get("data_type", "unknown"),
                    domain=domain
                )

                return {
                    "status": "inferred",
                    "method": "context_inference",
                    "score": inferred_concept["confidence"],
                    "concept": inferred_concept,
                    "reason": f"Inferred from {domain} domain context"
                }

        return result

    def reset_context(self):
        """Clear accumulated context (call when switching datasets)."""
        self.matched_columns = []
