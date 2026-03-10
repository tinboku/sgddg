"""Optimized LLM adapter with KG integration, multi-API load balancing, and cost optimization."""

import os
import json
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

# Load config
def load_config():
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()


from .kg_enhancer import KnowledgeGraphEnhancer


class SingleAPIClient:
    """Single API client for comparison experiments."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider_name = provider
        self.model_name = model
        self.stats = {'calls': 0, 'errors': 0, 'cost': 0.0}

        if provider == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
                self.provider_type = 'gemini'
                self.cost_per_1k = 0.0001
                print(f"Using Gemini API ({model})")
            except Exception as e:
                raise RuntimeError(f"Gemini init failed: {e}")

        elif provider == 'openai':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.provider_type = 'openai'
                self.cost_per_1k = 0.00015
                print(f"Using OpenAI API ({model})")
            except Exception as e:
                raise RuntimeError(f"OpenAI init failed: {e}")

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'gemini' or 'openai'")


    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate text with automatic retry."""
        for attempt in range(max_retries):
            try:
                if self.provider_type == 'gemini':
                    response = self.client.generate_content(prompt)
                    text = response.text

                    # Extract JSON
                    match = re.search(r"```(json)?\s*({.*?})\s*```", text, re.DOTALL)
                    if match:
                        text = match.group(2)

                    # Update stats
                    self.stats['calls'] += 1
                    self.stats['cost'] += self.cost_per_1k * (len(prompt) / 1000)

                    return text.strip()

                elif self.provider_type == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    text = response.choices[0].message.content

                    # Update stats
                    self.stats['calls'] += 1
                    self.stats['cost'] += self.cost_per_1k * (len(prompt) / 1000)

                    return text.strip()

            except Exception as e:
                self.stats['errors'] += 1
                print(f"Warning: {self.provider_name} call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        return ""

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'calls': self.stats['calls'],
            'errors': self.stats['errors'],
            'cost': self.stats['cost']
        }


class MultiAPIClient:
    """Multi-API client with load balancing and cost optimization."""

    def __init__(self, providers_config: List[Dict[str, Any]]):
        self.providers = []
        self.stats = {}

        for config in providers_config:
            provider_name = config['provider']
            model_name = config['model']
            api_key = config['api_key']
            cost_per_1k = config['cost_per_1k']

            if provider_name == 'gemini':
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    client = genai.GenerativeModel(model_name)
                    self.providers.append({
                        'name': 'gemini',
                        'type': 'gemini',
                        'model': model_name,
                        'client': client,
                        'cost_per_1k': cost_per_1k
                    })
                    self.stats['gemini'] = {'calls': 0, 'errors': 0, 'cost': 0.0}
                    print(f"Loaded Gemini API ({model_name})")
                except Exception as e:
                    print(f"Warning: Gemini init failed: {e}")

            elif provider_name == 'openai':
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    self.providers.append({
                        'name': 'openai',
                        'type': 'openai',
                        'model': model_name,
                        'client': client,
                        'cost_per_1k': cost_per_1k
                    })
                    self.stats['openai'] = {'calls': 0, 'errors': 0, 'cost': 0.0}
                    print(f"Loaded OpenAI API ({model_name})")
                except Exception as e:
                    print(f"Warning: OpenAI init failed: {e}")

        if not self.providers:
            print("Error: No API providers available!")

    def select_provider(self, prefer_cheap: bool = True):
        """Select API provider (cheapest first by default)."""
        if not self.providers:
            return None

        if prefer_cheap:
            return min(self.providers, key=lambda p: p['cost_per_1k'])
        else:
            import random
            return random.choice(self.providers)

    def generate(self, prompt: str, max_retries: int = 2, prefer_cheap: bool = True) -> str:
        """Generate text with automatic retry and failover."""
        for attempt in range(max_retries):
            provider = self.select_provider(prefer_cheap)
            if not provider:
                return ""

            try:
                if provider['type'] == 'gemini':
                    response = provider['client'].generate_content(prompt)
                    text = response.text

                    # Extract JSON
                    match = re.search(r"```(json)?\s*({.*?})\s*```", text, re.DOTALL)
                    if match:
                        text = match.group(2)

                    # Update stats
                    self.stats[provider['name']]['calls'] += 1
                    self.stats[provider['name']]['cost'] += provider['cost_per_1k'] * (len(prompt) / 1000)

                    return text.strip()

                elif provider['type'] == 'openai':
                    response = provider['client'].chat.completions.create(
                        model=provider['model'],
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    text = response.choices[0].message.content

                    # Update stats
                    self.stats[provider['name']]['calls'] += 1
                    self.stats[provider['name']]['cost'] += provider['cost_per_1k'] * (len(prompt) / 1000)

                    return text.strip()

            except Exception as e:
                self.stats[provider['name']]['errors'] += 1
                print(f"Warning: {provider['name']} call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

        return ""

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return self.stats


class OptimizedLLMAdapter:
    """LLM adapter with KG enhancement, multi-API support, parallel batching, and caching."""

    def __init__(self,
                 gemini_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 use_kg: bool = True,
                 prefer_cheap: bool = True,
                 enable_cache: bool = True):

        # Initialize multi-API client
        providers_config = []

        if gemini_key:
            providers_config.append({
                'provider': 'gemini',
                'model': 'gemini-2.5-flash',
                'api_key': gemini_key,
                'cost_per_1k': 0.0001
            })

        if openai_key:
            providers_config.append({
                'provider': 'openai',
                'model': 'gpt-4o-mini',
                'api_key': openai_key,
                'cost_per_1k': 0.00015
            })

        self.client = MultiAPIClient(providers_config) if providers_config else None

        # KG enhancer
        self.kg_enhancer = KnowledgeGraphEnhancer() if use_kg else None

        # Config
        self.prefer_cheap = prefer_cheap
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None

    def _get_cache_key(self, column_profile: Dict[str, Any]) -> str:
        """Generate cache key."""
        key_str = json.dumps(column_profile, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _build_enhanced_prompt(self, column_profile: Dict[str, Any]) -> str:
        """Build KG-enhanced prompt."""
        column_name = column_profile.get('column_name', 'N/A')
        data_type = column_profile.get('data_type', 'N/A')
        sample_values = ", ".join(map(str, column_profile.get('sample_values', [])))

        # Base prompt
        prompt = f"""You are an expert data analyst. Generate a clear, concise description for this dataset column.

**Column Information:**
- **Column Name:** `{column_name}`
- **Data Type:** `{data_type}`
- **Sample Values:** `{sample_values}`
"""

        # KG enhancement
        if self.kg_enhancer:
            concepts = self.kg_enhancer.search_concepts(column_name, top_k=2)
            if concepts:
                prompt += "\n**Related Domain Knowledge:**\n"
                for concept in concepts:
                    prompt += f"- **{concept['name']}**: {concept['definition'][:100]}...\n"

        prompt += """
**Task:** Generate a one-sentence description focusing on business meaning.

**Output Format (JSON):**
```json
{
  "description": "Your description here",
  "domain": "Detected domain (e.g., finance, healthcare, climate)",
  "confidence": 0.95
}
```
"""

        return prompt.strip()

    def generate_description(self, column_profile: Dict[str, Any]) -> str:
        """Generate metadata description for a single column."""
        if not self.client:
            return f"Description for '{column_profile.get('column_name')}' (no API configured)"

        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key(column_profile)
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Build prompt
        if "prompt" in column_profile:
            prompt = column_profile["prompt"]
        else:
            prompt = self._build_enhanced_prompt(column_profile)

        # Call LLM
        description = self.client.generate(prompt, prefer_cheap=self.prefer_cheap)

        # Cache result
        if self.enable_cache and description:
            self.cache[cache_key] = description

        return description if description else f"Generated description for '{column_profile.get('column_name')}'"

    def generate_descriptions_batch(self,
                                   column_profiles: List[Dict[str, Any]],
                                   max_workers: int = 5) -> List[str]:
        """Generate descriptions in parallel."""
        if not self.client:
            return [f"Description for column {i}" for i in range(len(column_profiles))]

        results = [None] * len(column_profiles)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_description, profile): idx
                for idx, profile in enumerate(column_profiles)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Batch generation failed (column {idx}): {e}")
                    results[idx] = f"Error generating description"

        return results

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        if self.client:
            return self.client.get_stats()
        return {}


class LLMAdapter(OptimizedLLMAdapter):
    """Backward-compatible adapter using the optimized version."""

    def __init__(self, api_key: Optional[str] = None):
        gemini_key = api_key or os.getenv('GEMINI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')

        super().__init__(
            gemini_key=gemini_key,
            openai_key=openai_key,
            use_kg=True,
            prefer_cheap=True,
            enable_cache=True
        )


if __name__ == "__main__":
    adapter = OptimizedLLMAdapter(
        gemini_key=os.getenv('GEMINI_API_KEY'),
        openai_key=os.getenv('OPENAI_API_KEY'),
        use_kg=True,
        prefer_cheap=True
    )

    # Single column test
    test_profile = {
        'column_name': 'temperature',
        'data_type': 'float',
        'sample_values': [23.5, 24.1, 22.8, 25.3]
    }

    description = adapter.generate_description(test_profile)
    print(f"Description: {description}")

    # Batch test
    test_profiles = [
        {'column_name': 'revenue', 'data_type': 'float', 'sample_values': [1000, 2000, 1500]},
        {'column_name': 'customer_id', 'data_type': 'int', 'sample_values': [1, 2, 3, 4]},
        {'column_name': 'date', 'data_type': 'datetime', 'sample_values': ['2024-01-01', '2024-01-02']}
    ]

    descriptions = adapter.generate_descriptions_batch(test_profiles, max_workers=3)
    for i, desc in enumerate(descriptions):
        print(f"Column {i}: {desc}")

    # Stats
    print("\nStats:")
    print(json.dumps(adapter.get_stats(), indent=2))
