#!/usr/bin/env python3
"""Column Relationship Analyzer - detects hierarchies, foreign keys, and dependencies between columns."""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict


class ColumnRelationshipAnalyzer:
    """Detects hierarchies, foreign keys, and functional dependencies between dataset columns."""

    # Common patterns for identifying different relationship types
    HIERARCHICAL_PATTERNS = {
        "geographic": ["continent", "country", "region", "state", "province", "city", "district", "zip"],
        "temporal": ["year", "quarter", "month", "week", "day", "hour"],
        "organizational": ["company", "department", "division", "team", "employee"],
        "product": ["category", "subcategory", "product", "sku", "variant"],
    }

    ID_PATTERNS = [
        r"^.*_id$",
        r"^.*_key$",
        r"^id_.*$",
        r"^key_.*$",
        r"^.*_code$",
        r"^.*_num(ber)?$",
    ]

    FINANCIAL_DEPENDENCIES = {
        # Pattern: (derived, [inputs])
        "revenue": ["sales", "income", "price", "quantity"],
        "profit": ["revenue", "cost", "expense"],
        "margin": ["profit", "revenue"],
        "total": ["subtotal", "tax", "shipping", "discount"],
        "net": ["gross", "tax", "deduction"],
    }

    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize the relationship analyzer.

        Args:
            min_confidence: Minimum confidence score to report a relationship
        """
        self.min_confidence = min_confidence

    def detect_relationships(
        self,
        df: pd.DataFrame,
        column_profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect all types of relationships in the dataset.

        Args:
            df: The dataset DataFrame
            column_profiles: List of column profile dictionaries

        Returns:
            Dictionary containing detected hierarchies, foreign keys, and dependencies
        """
        print("   -> Analyzing column relationships...")

        hierarchies = self._detect_hierarchies(df, column_profiles)
        foreign_keys = self._detect_foreign_keys(df, column_profiles)
        dependencies = self._detect_dependencies(df, column_profiles)

        result = {
            "hierarchies": hierarchies,
            "foreign_keys": foreign_keys,
            "dependencies": dependencies,
            "summary": {
                "total_hierarchies": len(hierarchies),
                "total_foreign_keys": len(foreign_keys),
                "total_dependencies": len(dependencies),
            }
        }

        print(f"      - Detected {len(hierarchies)} hierarchies")
        print(f"      - Detected {len(foreign_keys)} foreign key candidates")
        print(f"      - Detected {len(dependencies)} dependencies")

        return result

    def _detect_hierarchies(
        self,
        df: pd.DataFrame,
        column_profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect hierarchical relationships between columns.

        Uses two methods:
        1. Name-based detection (matching against known hierarchy patterns)
        2. Data-based detection (cardinality analysis)
        """
        hierarchies = []
        column_names = df.columns.tolist()

        # Method 1: Pattern-based detection
        for hierarchy_type, keywords in self.HIERARCHICAL_PATTERNS.items():
            matched_columns = []
            for keyword in keywords:
                for col_name in column_names:
                    if re.search(keyword, col_name.lower()):
                        matched_columns.append((col_name, keywords.index(keyword)))

            # Sort by hierarchy level and create hierarchy path
            if len(matched_columns) >= 2:
                matched_columns.sort(key=lambda x: x[1])  # Sort by level
                hierarchy_path = [col for col, _ in matched_columns]

                # Verify with data (parent should have fewer unique values than child)
                if self._verify_hierarchy_cardinality(df, hierarchy_path):
                    hierarchies.append({
                        "type": hierarchy_type,
                        "columns": hierarchy_path,
                        "confidence": 0.85,
                        "method": "pattern_based",
                        "description": f"{hierarchy_type.capitalize()} hierarchy detected"
                    })

        # Method 2: Data distribution-based detection
        # Look for columns where one's unique values are a subset of another
        cardinality_hierarchies = self._detect_cardinality_hierarchies(df, column_profiles)
        for hierarchy in cardinality_hierarchies:
            # Avoid duplicates from pattern-based detection
            if not any(set(h['columns']) == set(hierarchy['columns']) for h in hierarchies):
                hierarchies.append(hierarchy)

        return hierarchies

    def _verify_hierarchy_cardinality(
        self,
        df: pd.DataFrame,
        hierarchy_path: List[str]
    ) -> bool:
        """
        Verify that hierarchy follows cardinality rules:
        Parent should have fewer or equal unique values than child.
        """
        for i in range(len(hierarchy_path) - 1):
            parent = hierarchy_path[i]
            child = hierarchy_path[i + 1]

            if parent not in df.columns or child not in df.columns:
                return False

            parent_unique = df[parent].nunique()
            child_unique = df[child].nunique()

            # Parent should have fewer unique values (more general)
            if parent_unique > child_unique:
                return False

        return True

    def _detect_cardinality_hierarchies(
        self,
        df: pd.DataFrame,
        column_profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect hierarchies based on cardinality analysis.
        If column A's unique values are always associated with specific values in column B,
        there may be a hierarchical relationship.
        """
        hierarchies = []

        # Get categorical/low-cardinality columns
        categorical_cols = [
            p['column_name'] for p in column_profiles
            if p.get('cardinality_ratio', 1.0) < 0.5 and p.get('unique_count', 0) < 100
        ]

        # Compare pairs
        for i, col_a in enumerate(categorical_cols):
            for col_b in categorical_cols[i+1:]:
                if col_a not in df.columns or col_b not in df.columns:
                    continue

                # Check if col_a -> col_b (A is parent of B)
                is_parent = self._is_hierarchical_parent(df, col_a, col_b)
                if is_parent:
                    hierarchies.append({
                        "type": "data_inferred",
                        "columns": [col_a, col_b],
                        "confidence": 0.70,
                        "method": "cardinality_based",
                        "description": f"Each value in '{col_a}' maps to consistent values in '{col_b}'"
                    })

        return hierarchies

    def _is_hierarchical_parent(
        self,
        df: pd.DataFrame,
        parent_col: str,
        child_col: str
    ) -> bool:
        """
        Check if parent_col is a hierarchical parent of child_col.
        This is true if each parent value always maps to the same set of child values.
        """
        # Group by parent and check if child values are consistent
        grouped = df.groupby(parent_col)[child_col].apply(lambda x: x.nunique())

        # If most parent values map to only 1 child value, it's hierarchical
        single_mapping_ratio = (grouped == 1).sum() / len(grouped) if len(grouped) > 0 else 0

        # Also check cardinality (parent should have fewer unique values)
        parent_card = df[parent_col].nunique()
        child_card = df[child_col].nunique()

        return single_mapping_ratio > 0.8 and parent_card < child_card

    def _detect_foreign_keys(
        self,
        df: pd.DataFrame,
        column_profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect foreign key candidates.

        Uses two methods:
        1. Name pattern matching (*_id, *_key, etc.)
        2. Data characteristics (high uniqueness, non-null, specific data types)
        """
        foreign_keys = []

        for profile in column_profiles:
            col_name = profile['column_name']
            confidence = 0.0
            reasons = []

            # Check 1: Name pattern
            name_matches_pattern = any(
                re.match(pattern, col_name.lower()) for pattern in self.ID_PATTERNS
            )
            if name_matches_pattern:
                confidence += 0.4
                reasons.append("name_pattern_match")

            # Check 2: Data type (should be int, string, or mixed)
            data_type = profile.get('data_type', '')
            if data_type in ['integer', 'string']:
                confidence += 0.2
                reasons.append("appropriate_data_type")

            # Check 3: Uniqueness (foreign keys often have high cardinality)
            cardinality_ratio = profile.get('cardinality_ratio', 0)
            if cardinality_ratio > 0.3:  # Not primary key (which would be ~1.0)
                confidence += 0.2
                reasons.append("high_cardinality")

            # Check 4: Null rate (foreign keys typically allow nulls, but not too many)
            null_rate = profile.get('null_rate', 0)
            if null_rate < 0.3:
                confidence += 0.1
                reasons.append("low_null_rate")

            # Check 5: Look for matching column name in dataset (potential self-reference)
            if col_name.endswith('_id'):
                base_name = col_name[:-3]  # Remove '_id'
                if any(base_name in other_col.lower() for other_col in df.columns if other_col != col_name):
                    confidence += 0.1
                    reasons.append("potential_reference_column_exists")

            # Report if confidence exceeds threshold
            if confidence >= self.min_confidence:
                foreign_keys.append({
                    "column_name": col_name,
                    "confidence": round(confidence, 2),
                    "reasons": reasons,
                    "data_type": data_type,
                    "cardinality_ratio": round(cardinality_ratio, 3),
                    "null_rate": round(null_rate, 3),
                })

        return foreign_keys

    def _detect_dependencies(
        self,
        df: pd.DataFrame,
        column_profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect functional dependencies between columns.

        Uses two methods:
        1. Name-based inference (e.g., revenue/cost/profit relationships)
        2. Correlation analysis for numeric columns
        """
        dependencies = []

        # Method 1: Pattern-based financial dependencies
        for derived_keyword, input_keywords in self.FINANCIAL_DEPENDENCIES.items():
            # Find columns matching the derived term
            derived_cols = [
                col for col in df.columns
                if derived_keyword in col.lower()
            ]

            for derived_col in derived_cols:
                # Find potential input columns
                input_cols = []
                for input_keyword in input_keywords:
                    matching = [
                        col for col in df.columns
                        if input_keyword in col.lower() and col != derived_col
                    ]
                    input_cols.extend(matching)

                if input_cols:
                    # Verify with data if possible (for numeric columns)
                    verified = self._verify_dependency(df, derived_col, input_cols)

                    dependencies.append({
                        "from_columns": input_cols,
                        "to_column": derived_col,
                        "relation_type": derived_keyword,
                        "confidence": 0.75 if verified else 0.60,
                        "method": "pattern_based",
                        "verified": verified
                    })

        # Method 2: Correlation-based detection for numeric columns
        numeric_cols = [
            p['column_name'] for p in column_profiles
            if p.get('data_type') in ['integer', 'float'] and p['column_name'] in df.columns
        ]

        correlation_deps = self._detect_correlation_dependencies(df, numeric_cols)
        dependencies.extend(correlation_deps)

        return dependencies

    def _verify_dependency(
        self,
        df: pd.DataFrame,
        derived_col: str,
        input_cols: List[str]
    ) -> bool:
        """
        Verify if derived_col could be computed from input_cols.
        For numeric columns, check if values follow expected relationships.
        """
        # Ensure all columns are numeric
        try:
            all_cols = [derived_col] + input_cols
            numeric_data = df[all_cols].select_dtypes(include=[np.number])

            if len(numeric_data.columns) != len(all_cols):
                return False  # Not all columns are numeric

            # Simple verification: check if derived is similar to sum/difference/product of inputs
            if len(input_cols) == 2:
                col1, col2 = input_cols[0], input_cols[1]

                # Check sum relationship
                sum_diff = abs(df[derived_col] - (df[col1] + df[col2])).mean()
                sum_ratio = sum_diff / (df[derived_col].abs().mean() + 1e-10)

                # Check difference relationship
                diff_diff = abs(df[derived_col] - (df[col1] - df[col2])).mean()
                diff_ratio = diff_diff / (df[derived_col].abs().mean() + 1e-10)

                # Check product relationship
                prod_diff = abs(df[derived_col] - (df[col1] * df[col2])).mean()
                prod_ratio = prod_diff / (df[derived_col].abs().mean() + 1e-10)

                # If any relationship holds strongly, consider verified
                return any(ratio < 0.1 for ratio in [sum_ratio, diff_ratio, prod_ratio])

        except Exception:
            pass

        return False

    def _detect_correlation_dependencies(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect dependencies based on high correlation between numeric columns.
        """
        dependencies = []

        if len(numeric_cols) < 2:
            return dependencies

        try:
            # Compute correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Find high correlations (> 0.85 or < -0.85)
            for i, col_a in enumerate(numeric_cols):
                for col_b in numeric_cols[i+1:]:
                    corr_value = corr_matrix.loc[col_a, col_b]

                    if abs(corr_value) >= 0.85 and not np.isnan(corr_value):
                        dependencies.append({
                            "from_columns": [col_a],
                            "to_column": col_b,
                            "relation_type": "correlation",
                            "correlation": round(float(corr_value), 3),
                            "confidence": min(0.95, abs(corr_value)),
                            "method": "correlation_based",
                            "verified": True
                        })

        except Exception as e:
            print(f"      - Warning: Correlation analysis failed: {e}")

        return dependencies

    def get_column_relationships_summary(
        self,
        relationships: Dict[str, Any],
        column_name: str
    ) -> Dict[str, Any]:
        """
        Extract relationship information for a specific column.

        Args:
            relationships: The full relationship dictionary from detect_relationships()
            column_name: The column to get relationships for

        Returns:
            Dictionary with parent, children, related, and dependencies for this column
        """
        summary = {
            "parent": None,
            "children": [],
            "related": [],
            "dependencies": {
                "depends_on": [],  # Columns this column depends on
                "depended_by": []  # Columns that depend on this column
            }
        }

        # Find parent/children from hierarchies
        for hierarchy in relationships.get('hierarchies', []):
            columns = hierarchy['columns']
            if column_name in columns:
                idx = columns.index(column_name)
                if idx > 0:
                    summary['parent'] = columns[idx - 1]
                if idx < len(columns) - 1:
                    summary['children'] = columns[idx + 1:]

        # Find related columns from foreign keys
        for fk in relationships.get('foreign_keys', []):
            if fk['column_name'] == column_name:
                # This column references another entity
                base_name = column_name.replace('_id', '').replace('_key', '')
                potential_related = [
                    col for col in relationships.get('hierarchies', [])
                    if base_name in str(col)
                ]
                if potential_related:
                    summary['related'].extend(potential_related)

        # Find dependencies
        for dep in relationships.get('dependencies', []):
            if dep['to_column'] == column_name:
                summary['dependencies']['depends_on'] = dep['from_columns']
            if column_name in dep['from_columns']:
                summary['dependencies']['depended_by'].append(dep['to_column'])

        return summary


# Demo/Test
if __name__ == "__main__":
    print("ColumnRelationshipAnalyzer Demo\n")

    # Create sample dataset with known relationships
    df = pd.DataFrame({
        "Country": ["USA", "USA", "China", "China", "USA", "China"],
        "State": ["CA", "NY", "Beijing", "Shanghai", "CA", "Beijing"],
        "City": ["LA", "NYC", "Beijing City", "Shanghai City", "SF", "Beijing City"],
        "customer_id": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "order_id": ["O001", "O002", "O003", "O004", "O005", "O006"],
        "Revenue": [100, 200, 150, 180, 220, 160],
        "Cost": [60, 120, 90, 110, 130, 95],
        "Profit": [40, 80, 60, 70, 90, 65],
    })

    # Create mock column profiles
    from profiling.column_profiler import ColumnProfiler
    profiler = ColumnProfiler()
    column_profiles = [
        profiler.profile(df[col], col) for col in df.columns
    ]

    # Analyze relationships
    analyzer = ColumnRelationshipAnalyzer(min_confidence=0.6)
    relationships = analyzer.detect_relationships(df, column_profiles)

    print("\n=== Detected Relationships ===")
    print(f"\nHierarchies ({len(relationships['hierarchies'])}):")
    for h in relationships['hierarchies']:
        print(f"  - {h['type']}: {' -> '.join(h['columns'])} (confidence: {h['confidence']})")

    print(f"\nForeign Keys ({len(relationships['foreign_keys'])}):")
    for fk in relationships['foreign_keys']:
        print(f"  - {fk['column_name']} (confidence: {fk['confidence']}, reasons: {fk['reasons']})")

    print(f"\nDependencies ({len(relationships['dependencies'])}):")
    for dep in dependencies:
        print(f"  - {' + '.join(dep['from_columns'])} -> {dep['to_column']} ({dep['relation_type']})")

    print("\n=== Column-specific Summary (Profit) ===")
    profit_summary = analyzer.get_column_relationships_summary(relationships, "Profit")
    print(f"Depends on: {profit_summary['dependencies']['depends_on']}")
    print(f"Parent: {profit_summary['parent']}")

    print("\nDemo complete!")
