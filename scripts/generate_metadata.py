#!/usr/bin/env python3
"""Generate metadata for a dataset using the SGDDG pipeline.

Usage:
    python scripts/generate_metadata.py --input data/my_dataset.csv --name "My Dataset"
    python scripts/generate_metadata.py --input data/my_dataset.csv --output results/output.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgddg.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate metadata for a CSV dataset")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--name", default=None, help="Dataset name (defaults to filename)")
    parser.add_argument("--kg-dir", default="data", help="KG data directory")
    parser.add_argument("--output", default=None, help="Output JSON path (optional)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    args = parser.parse_args()

    dataset_name = args.name or Path(args.input).stem

    result = run_pipeline(
        dataset_path=args.input,
        kg_data_dir=args.kg_dir,
        dataset_name=dataset_name,
        enable_parallel=not args.no_parallel,
        enable_cache=not args.no_cache,
    )

    # Display results
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    print(f"\n--- UFD ---")
    print(f"Title: {result.ufd.get('title', 'N/A')}")
    print(result.ufd.get("core_description", {}).get("text", "No description generated."))

    print(f"\n--- SFD ---")
    sfd = result.sfd
    print(f"Keywords: {len(sfd.get('keywords', {}).get('core', []))} core, "
          f"{len(sfd.get('keywords', {}).get('related', []))} related")
    print(f"Summary: {sfd.get('summary', 'N/A')}")

    print(f"\n--- Stats ---")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")

    # Save to file if requested
    if args.output:
        output = {
            "dataset_name": dataset_name,
            "ufd": result.ufd,
            "sfd": result.sfd,
            "stats": result.stats,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
