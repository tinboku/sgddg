#!/usr/bin/env python3
"""Build the concept dictionary from domain lexicon files.

Scans a directory for JSON/CSV lexicon files and populates the
knowledge graph with concepts, aliases, and embeddings.

Usage:
    python scripts/build_kg.py --lexicon-dir data/lexicons --output data/schema_kg.db
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiling.schema_extractor import SchemaExtractor
from kg.kg_manager import KnowledgeGraphManager

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def parse_lexicon_entry(
    term_key: str,
    term_data: Dict[str, Any],
    source_file: str
) -> Tuple[Dict, List]:
    """Parse a single lexicon entry into concept and alias records."""
    raw_def = term_data.get("definition")
    if isinstance(raw_def, dict):
        definition = raw_def.get("en")
    else:
        definition = raw_def

    canonical_name = term_data.get("canonical_name", term_key)
    display_name = term_data.get("display_name") or canonical_name or term_key

    concept = {
        "concept_id": term_key.lower().replace(" ", "_"),
        "canonical_name": canonical_name,
        "display_name": display_name,
        "definition": definition or "",
        "domain": term_data.get("domain", "general"),
        "source": source_file,
    }

    aliases = []
    for alias in term_data.get("aliases", []):
        aliases.append({
            "alias": alias,
            "concept_id": concept["concept_id"],
            "source": source_file,
        })

    return concept, aliases


def build_from_directory(lexicon_dir: str, output_db: str):
    """Scan lexicon directory and build the knowledge graph."""
    lexicon_path = Path(lexicon_dir)
    if not lexicon_path.exists():
        print(f"Error: lexicon directory not found: {lexicon_dir}")
        sys.exit(1)

    kg = KnowledgeGraphManager(kg_directory=str(Path(output_db).parent))

    json_files = list(lexicon_path.glob("*.json"))
    print(f"Found {len(json_files)} lexicon files in {lexicon_dir}")

    total_concepts = 0
    total_aliases = 0

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            lexicon = json.load(f)

        for term_key, term_data in lexicon.items():
            concept, aliases = parse_lexicon_entry(
                term_key, term_data, json_file.name
            )
            kg.add_concept(concept)
            for alias in aliases:
                kg.add_alias(alias)
            total_concepts += 1
            total_aliases += len(aliases)

    print(f"Built KG: {total_concepts} concepts, {total_aliases} aliases")

    if EMBEDDINGS_AVAILABLE:
        print("Generating embeddings...")
        kg.build_vector_index()
        print("Vector index built.")
    else:
        print("Warning: sentence-transformers not installed, skipping embeddings")


def main():
    parser = argparse.ArgumentParser(description="Build SGDDG concept dictionary")
    parser.add_argument("--lexicon-dir", default="data/lexicons",
                        help="Directory containing JSON lexicon files")
    parser.add_argument("--output", default="data/schema_kg.db",
                        help="Output database path")
    args = parser.parse_args()

    build_from_directory(args.lexicon_dir, args.output)


if __name__ == "__main__":
    main()
