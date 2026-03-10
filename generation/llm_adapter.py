"""LLM Adapter - unified LLM interface routing calls to OpenAI or Gemini."""
import os
import json
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def load_config():
    """Load config.yaml from project root."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class LLMClient:
    """Unified LLM client supporting multiple providers (OpenAI, Gemini)."""

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.provider = provider
        self.model_name = model
        self.api_key = api_key
        self.client = None
        self._config = config or {}
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate API client based on provider."""
        if self.provider == "openai":
            if not self.api_key:
                print("Warning: OpenAI API key not provided.")
                return
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"OpenAI client initialized (model: {self.model_name}).")
            except ImportError:
                print("Error: 'openai' package not installed. Run: pip install openai")
            except Exception as e:
                print(f"Error: OpenAI client initialization failed: {e}")

        elif self.provider == "gemini":
            if not self.api_key:
                print("Warning: Gemini API key not provided.")
                return
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print(f"Gemini client initialized (model: {self.model_name}).")
            except ImportError:
                print("Error: 'google-genai' not installed. Run: pip install google-genai")
            except Exception as e:
                print(f"Error: Gemini client initialization failed: {e}")

    def generate(self, prompt: str) -> str:
        """Call the LLM to generate text."""
        if not self.client:
            print("   (LLM client not initialized, returning empty string)")
            return ""

        gen_config = self._config.get("generation", {})
        temperature = gen_config.get("temperature", 0.0)

        if self.provider == "openai":
            return self._generate_openai(prompt, temperature)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, temperature)

        return ""

    def _generate_openai(self, prompt: str, temperature: float) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            raw_response = response.choices[0].message.content or ""
            return self._extract_json_if_present(raw_response)
        except Exception as e:
            print(f"Error: OpenAI API call failed: {e}")
            return ""

    def _generate_gemini(self, prompt: str, temperature: float) -> str:
        """Generate text using Gemini API."""
        try:
            from google.genai import types

            config_args = {}
            if temperature is not None:
                config_args["temperature"] = temperature

            generation_config = types.GenerateContentConfig(**config_args) if config_args else None

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )
            raw_response = response.text
            return self._extract_json_if_present(raw_response)
        except Exception as e:
            print(f"Error: Gemini API call failed: {e}")
            return ""

    @staticmethod
    def _extract_json_if_present(text: str) -> str:
        """Extract JSON from markdown code blocks if present."""
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Also check for JSON arrays
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return text


class LLMAdapter:
    """High-level adapter for metadata generation LLM calls with config-based key resolution."""
    def __init__(self, api_key: Optional[str] = None):
        config = load_config()
        llm_provider = config["generation"].get("llm_provider", "openai")
        llm_model = config["generation"].get("llm_model", "gpt-4o-mini")

        # Resolve API key: explicit > config > environment variable
        if api_key:
            resolved_key = api_key
        elif llm_provider == "openai":
            resolved_key = config.get("api_keys", {}).get("openai") or os.getenv("OPENAI_API_KEY")
        elif llm_provider == "gemini":
            resolved_key = config.get("api_keys", {}).get("gemini") or os.getenv("GEMINI_API_KEY")
        else:
            resolved_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

        self.client = LLMClient(
            provider=llm_provider,
            model=llm_model,
            api_key=resolved_key,
            config=config
        )
        self._config = config

    def _build_prompt(self, column_profile: Dict[str, Any]) -> str:
        """Build a high-quality prompt from column profile information."""
        column_name = column_profile.get('column_name', 'N/A')
        data_type = column_profile.get('data_type', 'N/A')
        sample_values = ", ".join(map(str, column_profile.get('sample_values', [])))

        prompt = f"""
You are an expert data analyst. Your task is to write a clear, concise, and business-oriented description for a dataset column.

**Column Information:**
- **Column Name:** `{column_name}`
- **Data Type:** `{data_type}`
- **Sample Values:** `{sample_values}`

**Instructions:**
Based on the information above, please generate a one-sentence description of what this column represents. Focus on the business meaning.

**Example:**
- **Column Name:** `transaction_amount`
- **Description:** Represents the total amount of money exchanged in a single transaction.

**Your turn:**
- **Column Name:** `{column_name}`
- **Description:**
"""
        return prompt.strip()

    def generate_description(self, column_profile: Dict[str, Any]) -> str:
        """Generate a metadata description from column profile or a direct prompt dict."""
        if not self.client:
            return f"AI description for '{column_profile.get('column_name')}' (LLM client not configured)."

        # Use direct prompt if provided; otherwise build from column profile
        if "prompt" in column_profile:
            prompt = column_profile["prompt"]
        else:
            prompt = self._build_prompt(column_profile)

        description = self.client.generate(prompt)

        if not description:
            return f"Generated description for '{column_profile.get('column_name')}' based on sample values."

        return description.strip()
