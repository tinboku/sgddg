"""UFD Generator - produces user-facing natural language descriptions for datasets."""
import json
from typing import Dict, Any, List, Optional

from .llm_adapter import LLMAdapter

class UFDGenerator:
    """Generates a User-Facing Description (UFD) leveraging rich semantic profiles."""

    def __init__(self, api_key: Optional[str] = None):
        self.llm_adapter = LLMAdapter(api_key=api_key)

    def _build_prompt(self, dataset_name: str, columns_data: List[Dict[str, Any]]) -> str:
        """Build a prompt for UFD generation incorporating semantic profiles."""

        column_summaries = []
        for col in columns_data:
            summary = {
                "column_name": col.get("column_name"),
                "concept": col.get("match_info", {}).get("concept", {}).get("display_name", "N/A"),
                "semantic_profile": col.get("semantic_profile")
            }
            column_summaries.append(summary)

        prompt = f"""
You are an expert data cataloger. Your task is to write a clear, concise, and engaging user-facing description for a dataset. The description should be a single, fluent paragraph that synthesizes the information from all columns into a coherent narrative about the dataset's purpose and content.

**Dataset Name:**
{dataset_name}

**Column Analysis (Name, Matched Concept, Semantic Profile):**
```json
{json.dumps(column_summaries, indent=2)}
```

**Instructions:**
- Analyze the provided JSON which details the columns, their matched real-world concepts, and their semantic profiles (Temporal, Spatial, EntityType, Domain, FunctionalRole).
- Synthesize this information to understand the dataset's overall theme, scope, and potential use cases.
- Write a single, compelling paragraph that describes the dataset to a general business audience.
- **Do not** simply list the columns. Instead, weave them into a narrative. For example, instead of saying "It has columns for country, year, and GDP," say "This dataset tracks the annual Gross Domestic Product (GDP) for various countries over time."

**Example Output:**
"This dataset provides a comprehensive overview of financial and ESG (Environmental, Social, and Governance) performance for a range of companies. It captures key financial metrics such as total revenue and net income, alongside critical ESG indicators like carbon emissions and employee turnover rates. The data is ideal for investors and analysts looking to perform integrated analysis of corporate financial health and sustainability practices."

**Your turn:**
Generate a single-paragraph description for the dataset:
"""
        return prompt.strip()

    def generate(self, dataset_name: str, columns_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate the UFD."""
        prompt = self._build_prompt(dataset_name, columns_data)

        print("   -> Generating User-Facing Description (UFD) via LLM...")
        description_text = self.llm_adapter.generate_description({"prompt": prompt})

        # Extract tags from semantic profiles
        tags = set()
        for col in columns_data:
            if col.get("semantic_profile"):
                tags.add(col["semantic_profile"].get("EntityType"))
                tags.add(col["semantic_profile"].get("Domain"))

        ufd_output = {
            "title": dataset_name.replace("_", " ").title(),
            "core_description": {
                "text": description_text or "A description could not be generated.",
                "format": "markdown"
            },
            "tags": sorted([tag for tag in tags if tag]) # Remove None and sort
        }

        print("   -> UFD generation complete.")
        return ufd_output


if __name__ == "__main__":
    print("UFDGenerator Demo")

    generator = UFDGenerator()
    print("UFD generator initialized")

    dataset_name = "financial_performance.csv"
    columns_data = [
        {"column_name": "Year", "match_info": {"concept": {"display_name": "reporting_year"}}, "semantic_profile": {"EntityType": "Year", "Domain": "general"}},
        {"column_name": "Revenue", "match_info": {"concept": {"display_name": "total_revenue"}}, "semantic_profile": {"EntityType": "Revenue", "Domain": "finance"}},
        {"column_name": "Profit", "match_info": {"concept": {"display_name": "net_profit"}}, "semantic_profile": {"EntityType": "Profit", "Domain": "finance"}}
    ]

    print(f"\nGenerating UFD...")
    ufd = generator.generate(dataset_name, columns_data)

    print(f"\nUFD Result:")
    print(f"  Title: {ufd['title']}")
    print(f"  Description: {ufd['core_description']['text'][:150]}...")
    print(f"  Tags: {ufd.get('tags', [])}")

    print("\nDemo complete!")
