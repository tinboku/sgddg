"""Schema Extractor - auto-extracts field dictionaries (concepts, aliases, evidence) from CSV files."""

import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime

from .column_profiler import ColumnProfiler


class SchemaExtractor:
    """Automatic schema extractor from CSV files."""

    def __init__(self):
        self.profiler = ColumnProfiler()

    def extract_from_csv(
        self,
        file_path: str,
        dataset_id: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Extract schema from a CSV file, returning concepts, aliases, and evidence."""
        # Read data
        df = pd.read_csv(file_path)

        # Generate dataset ID
        if not dataset_id:
            file_name = Path(file_path).stem
            dataset_id = self._generate_dataset_id(file_name)

        # Extract concepts
        concepts = []
        aliases = []
        evidences = []

        for column_name in df.columns:
            # Profile column
            profile = self.profiler.profile(df[column_name], column_name)

            # Generate concept
            concept = self._profile_to_concept(profile, dataset_id)
            concepts.append(concept)

            # Generate aliases
            column_aliases = self._extract_aliases(column_name, concept["id"])
            aliases.extend(column_aliases)

            # Generate evidence
            evidence = self._profile_to_evidence(profile, concept["id"], dataset_id)
            evidences.append(evidence)

        return {
            "dataset_id": dataset_id,
            "dataset_name": Path(file_path).name,
            "file_path": file_path,
            "extraction_time": datetime.now().isoformat(),
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "concepts": concepts,
            "aliases": aliases,
            "evidences": evidences
        }

    def _generate_dataset_id(self, file_name: str) -> str:
        """Generate dataset ID from file name hash."""
        hash_obj = hashlib.md5(file_name.encode())
        hash_hex = hash_obj.hexdigest()[:8]
        return f"ds_{hash_hex}"

    def _profile_to_concept(self, profile: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
        """Convert column profile to concept dictionary."""
        column_name = profile["column_name"]

        # Normalize name
        canonical_name = self._normalize_name(column_name)

        # Generate concept ID
        concept_id = f"{canonical_name}_{dataset_id}"

        # Infer concept type
        semantic_type = profile["inferred_semantic_type"]
        concept_type = self._semantic_to_concept_type(semantic_type)

        # Infer domain
        domain = self._infer_domain(semantic_type, column_name)

        # Generate definition
        definition = self._generate_definition(column_name, profile)

        # Attributes
        attributes = {
            "unit": profile["unit"],
            "data_type": profile["data_type"],
        }

        if profile["statistics"]:
            if "min" in profile["statistics"] and "max" in profile["statistics"]:
                attributes["range"] = {
                    "min": profile["statistics"]["min"],
                    "max": profile["statistics"]["max"]
                }

        return {
            "id": concept_id,
            "canonical_name": canonical_name,
            "concept_type": concept_type,
            "domain": domain,
            "definition": definition,
            "attributes": attributes,
            "tags": [semantic_type, domain, concept_type],
            "confidence": 0.8,
            "created_at": datetime.now().isoformat(),
            "source": dataset_id
        }

    def _normalize_name(self, name: str) -> str:
        """Normalize name: remove special chars, lowercase, underscore-join."""
        import re
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        name = name.lower()
        return name

    def _semantic_to_concept_type(self, semantic_type: str) -> str:
        """Map semantic type to concept type."""
        mapping = {
            "metric": "metric",
            "ratio": "metric",
            "count": "metric",
            "numerical": "numerical",
            "identifier": "identifier",
            "temporal": "temporal",
            "spatial": "spatial",
            "categorical": "categorical",
            "general": "dimension"
        }
        return mapping.get(semantic_type, "dimension")

    def _infer_domain(self, semantic_type: str, column_name: str) -> str:
        """Infer domain from semantic type and column name."""
        name_lower = column_name.lower()

        # Finance
        if any(kw in name_lower for kw in [
            "revenue", "profit", "sales", "income", "earnings", "cash",
            "asset", "liability", "equity", "dividend", "margin"
        ]):
            return "finance"

        # Healthcare
        if any(kw in name_lower for kw in [
            "patient", "diagnosis", "treatment", "symptom", "disease",
            "hospital", "doctor", "medicine", "health"
        ]):
            return "healthcare"

        # Education
        if any(kw in name_lower for kw in [
            "student", "grade", "course", "school", "teacher", "education"
        ]):
            return "education"

        return "general"

    def _generate_definition(self, column_name: str, profile: Dict[str, Any]) -> str:
        """Generate a definition string from column name and profile."""
        data_type = profile["data_type"]
        semantic_type = profile["inferred_semantic_type"]
        unit = profile["unit"]

        definition = f"A {semantic_type} field representing {column_name}"

        if unit:
            definition += f", measured in {unit}"

        definition += f". Data type: {data_type}."

        return definition

    def _extract_aliases(self, column_name: str, concept_id: str) -> List[Dict[str, Any]]:
        """Extract aliases from column name."""
        aliases = []

        # Original column name as alias
        aliases.append({
            "concept_id": concept_id,
            "alias_text": column_name,
            "alias_type": "exact_synonym",
            "confidence": 1.0,
            "created_at": datetime.now().isoformat(),
            "created_by": "auto_extraction"
        })

        # Generate common variants
        import re
        without_parens = re.sub(r'\([^)]*\)', '', column_name).strip()
        if without_parens != column_name and without_parens:
            aliases.append({
                "concept_id": concept_id,
                "alias_text": without_parens,
                "alias_type": "case_variant",
                "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "created_by": "auto_extraction"
            })

        return aliases

    def _profile_to_evidence(
        self,
        profile: Dict[str, Any],
        concept_id: str,
        dataset_id: str
    ) -> Dict[str, Any]:
        """Convert column profile to evidence dictionary."""
        return {
            "concept_id": concept_id,
            "dataset_id": dataset_id,
            "column_name": profile["column_name"],
            "evidence_type": "statistical_profile",
            "sample_values": profile["sample_values"],
            "statistical_profile": profile["statistics"],
            "semantic_profile": {
                "semantic_type": profile["inferred_semantic_type"],
                "data_type": profile["data_type"],
                "unit": profile["unit"]
            },
            "confidence": 0.8,
            "created_at": datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("SchemaExtractor Demo")

    import tempfile
    import os

    df = pd.DataFrame({
        "Year": [2020, 2021, 2022, 2023],
        "Revenue ($M)": [100.5, 120.3, 135.7, 150.2],
        "Operating Margin (%)": [15.2, 16.8, 17.5, 18.1],
        "Country": ["USA", "USA", "China", "China"]
    })

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    print(f"Created sample CSV: {temp_file.name}")

    extractor = SchemaExtractor()

    print(f"\nExtracting schema...")
    schema = extractor.extract_from_csv(temp_file.name)

    print(f"\nResults:")
    print(f"   Dataset ID: {schema['dataset_id']}")
    print(f"   Total columns: {schema['total_columns']}")
    print(f"   Total rows: {schema['total_rows']}")
    print(f"   Concepts: {len(schema['concepts'])}")
    print(f"   Aliases: {len(schema['aliases'])}")
    print(f"   Evidence: {len(schema['evidences'])}")

    print(f"\nSample concepts:")
    for concept in schema['concepts'][:2]:
        print(f"\n  - {concept['canonical_name']}")
        print(f"    Type: {concept['concept_type']}")
        print(f"    Domain: {concept['domain']}")
        print(f"    Definition: {concept['definition'][:60]}...")

    os.unlink(temp_file.name)

    print("\nDemo complete!")
