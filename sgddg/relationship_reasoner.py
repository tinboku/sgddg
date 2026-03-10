"""
Relationship reasoner for SGDDG.

Uses KG edges to discover semantic relationships between matched columns
and infer global dataset context (e.g., domain, common anchors, patterns).
"""

import sqlite3
import os
from typing import List, Dict, Any, Set, Optional
from collections import Counter

class RelationshipReasoner:
    """Infers dataset-level context from KG relationships between matched columns."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def infer_dataset_context(self, column_matches: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze matched columns to infer global dataset context.

        Args:
            column_matches: {column_name: match_result} mapping

        Returns:
            Dict with anchors, domain consensus, recognized pattern, and insight text.
        """
        if not os.path.exists(self.db_path):
            return {"error": "KG database not found"}

        concept_ids = []
        col_to_cid = {}

        # Collect concept IDs from successful matches
        for col, match in column_matches.items():
            if match.get("status") == "matched" and "concept" in match:
                cid = match["concept"].get("id")
                if cid:
                    concept_ids.append(cid)
                    col_to_cid[cid] = col

        if len(concept_ids) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 matched columns for reasoning"}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find direct relationships between matched concepts
        placeholders = ','.join(['?'] * len(concept_ids))
        query = f"""
            SELECT r.source_concept_id, r.relationship_type, r.target_concept_id,
                   s.display_name, t.display_name
            FROM relationships r
            JOIN concepts s ON r.source_concept_id = s.id
            JOIN concepts t ON r.target_concept_id = t.id
            WHERE r.source_concept_id IN ({placeholders})
              AND r.target_concept_id IN ({placeholders})
        """

        cursor.execute(query, concept_ids + concept_ids)
        direct_links = cursor.fetchall()

        # Find common anchor nodes (concepts pointed to by multiple matches)
        anchor_query = f"""
            SELECT t.display_name, COUNT(*) as link_count
            FROM relationships r
            JOIN concepts t ON r.target_concept_id = t.id
            WHERE r.source_concept_id IN ({placeholders})
            GROUP BY r.target_concept_id
            HAVING link_count >= 2
            ORDER BY link_count DESC
            LIMIT 3
        """
        cursor.execute(anchor_query, concept_ids)
        top_anchors = cursor.fetchall()

        conn.close()

        # Domain consensus from match metadata
        domains = [match.get("concept", {}).get("domain") for match in column_matches.values()
                   if match.get("domain") and match.get("domain") != 'Uncategorized']
        if not domains:
            domains = [match.get("concept", {}).get("domain") for match in column_matches.values()
                       if match.get("status") == "matched" and match.get("concept", {}).get("domain")]

        domain_counts = Counter(domains)
        most_common_domain = domain_counts.most_common(1)

        # Pattern recognition
        dataset_pattern = self._recognize_dataset_pattern(list(column_matches.keys()))

        # Build results
        relationships_found = []
        for src_id, rel, tgt_id, src_name, tgt_name in direct_links:
            relationships_found.append(f"{src_name} --[{rel}]--> {tgt_name}")

        anchors = [{"name": name, "strength": count} for name, count in top_anchors]

        # Generate dataset insight (prefer recognized patterns)
        dataset_insight = ""

        if dataset_pattern:
            dataset_insight = dataset_pattern["insight"]
            print(f"    Recognized Pattern: {dataset_pattern['pattern_name']}")
        elif anchors:
            dataset_insight = f"This dataset is strongly centered around '{anchors[0]['name']}'."
        elif most_common_domain:
            dataset_insight = f"This dataset is primarily related to the '{most_common_domain[0][0]}' domain."

        if relationships_found:
            dataset_insight += f" Key internal relationships discovered: {'; '.join(relationships_found[:3])}."

        return {
            "status": "success",
            "anchors": anchors,
            "domain_consensus": most_common_domain[0][0] if most_common_domain else None,
            "pattern": dataset_pattern["pattern_name"] if dataset_pattern else None,
            "internal_links": relationships_found,
            "dataset_insight": dataset_insight
        }

    def _recognize_dataset_pattern(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        """Recognize common dataset patterns from column names."""
        normalized_cols = [col.lower().strip() for col in column_names]

        # Pattern: OHLC financial time series
        ohlc_indicators = {'open', 'high', 'low', 'close'}
        if ohlc_indicators.issubset(set(normalized_cols)):
            has_volume = 'volume' in normalized_cols
            has_ticker = any(t in normalized_cols for t in ['ticker', 'symbol', 'stock'])

            insight = "This is a financial OHLC (Open-High-Low-Close) time series dataset"
            if has_ticker:
                insight += ", tracking multiple securities identified by ticker symbols"
            if has_volume:
                insight += ", including trading volume data"
            insight += ". Commonly used for technical analysis, price prediction, and algorithmic trading strategies."

            return {
                "pattern_name": "Financial_OHLC_TimeSeries",
                "insight": insight
            }

        # Pattern: healthcare insurance underwriting
        insurance_indicators = {'charges', 'premium', 'bmi', 'smoker', 'age'}
        if len(insurance_indicators.intersection(set(normalized_cols))) >= 3:
            insight = "This is a healthcare actuarial dataset for insurance risk assessment and premium calculation. "
            insight += "It combines demographic factors (age, region), health metrics (BMI), and lifestyle indicators (smoking status) "
            insight += "to model medical insurance costs. Ideal for predictive modeling of healthcare expenses and policy pricing optimization."

            return {
                "pattern_name": "Healthcare_Insurance_Underwriting",
                "insight": insight
            }

        # Pattern: e-commerce transactions
        ecommerce_indicators = {'order_id', 'customer_id', 'product', 'quantity', 'price'}
        if len(ecommerce_indicators.intersection(set(normalized_cols))) >= 3:
            return {
                "pattern_name": "E-commerce_Transaction",
                "insight": "This is an e-commerce transaction dataset capturing customer purchase behavior, product sales, and order details. Suitable for customer segmentation, product recommendation, and sales forecasting."
            }

        # Pattern: generic time series
        time_indicators = {'date', 'time', 'timestamp', 'year', 'month', 'day'}
        if any(t in normalized_cols for t in time_indicators):
            has_metrics = len([c for c in normalized_cols if c not in time_indicators]) >= 2
            if has_metrics:
                return {
                    "pattern_name": "Generic_TimeSeries",
                    "insight": "This is a time-indexed dataset suitable for temporal analysis, trend detection, and forecasting."
                }

        return None
