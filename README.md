# SGDDG: Schema-Guided Dataset Description Generator

> Master's thesis project (2025). Code extracted and uploaded for reference.

A metadata generation framework for data lake datasets, combining schema-level semantic analysis with knowledge graph-enhanced LLM generation.

## Overview

Data lakes accumulate tens of thousands of datasets, but their metadata is often incomplete, inconsistent, or semantically ambiguous. SGDDG addresses this by replacing row-level sampling with structured schema analysis and external domain knowledge injection.

The framework generates two types of metadata:
- **UFD (User-Facing Description)**: A readable narrative for data consumers
- **SFD (Search-Facing Description)**: A keyword-dense document optimized for dataset retrieval

## Architecture

```
CSV Dataset
    │
    ▼
┌─────────────────────────┐
│  1. Semantic Profiling   │  Physical stats, semantic types,
│     (column_profiler)    │  relationships, constraints
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  2. Schema Matching      │  BM25 + dense retrieval + RRF
│     (matchers, kg)       │  → concept dictionary (8,176 concepts)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  3. Conflict Resolution  │  Problem classification, adaptive routing,
│     (resolvers, filters) │  tier-based filtering, prompt compression
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  4. LLM Generation       │  Unified single-call generation
│     (unified_generator)  │  → UFD + SFD + semantic profiles
└─────────────────────────┘
```

## Project Structure

```
sgddg/                  Core pipeline
├── pipeline.py         Main orchestrator
├── unified_generator.py  Single-call metadata generation
├── cache.py            Profile and KG match caching
├── matchers/           Multi-granularity schema matching
├── classifiers/        Column problem type classification
├── filters/            Tier-based concept filtering
├── routers/            Confidence-based adaptive routing
├── reasoners/          Cross-column relationship reasoning
├── resolvers/          Statistical conflict resolution
└── compressors/        Prompt compression

profiling/              Semantic profiling modules
├── column_profiler.py  Physical and statistical analysis
├── schema_matcher.py   Cascaded matching (exact → fuzzy → semantic)
├── reranker.py         Cross-encoder reranking
└── bm25_index.py       BM25 index construction

generation/             LLM generation adapters
├── llm_adapter.py      Base LLM client (OpenAI/Gemini)
├── ufd_generator.py    User-facing description generation
├── sfd_generator.py    Search-facing description generation
└── kg_enhancer.py      Knowledge graph enhancement

kg/                     Knowledge graph management
├── kg_manager.py       Unified KG interface
├── concept_store.py    Concept storage (SQLite)
└── vector_store.py     Vector index (FAISS)

eval/                   Evaluation tools
├── ranking_metrics.py  NDCG, MAP, MRR computation
├── bm25_retriever.py   BM25 retrieval for evaluation
└── benchmark_runner.py Benchmark execution framework
```

## Setup

### Requirements

- Python 3.10+
- OpenAI API key (or Gemini API key)

### Installation

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys
```

Or set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
```

### Building the Concept Dictionary

```bash
python scripts/build_kg.py --domains medical financial geographic
```

### Running the Pipeline

```python
from sgddg import run_pipeline

result = run_pipeline(
    dataset_path="path/to/dataset.csv",
    dataset_name="My Dataset",
    kg_data_dir="data",
)

print(result.ufd)   # User-facing description
print(result.sfd)   # Search-facing description
print(result.stats)  # Timing and cache statistics
```

## Key Design Decisions

**Problem Classification**: Columns are classified into problem types (abbreviation, entity linking, domain anchoring, etc.) before KG lookup. This enables type-specific matching strategies rather than a one-size-fits-all approach.

**Multi-Granularity Matching**: Schema matching combines BM25 lexical search, dense semantic retrieval, and cross-encoder reranking via Reciprocal Rank Fusion, handling both exact terminology and semantic similarity.

**Tier-Based Filtering**: Matched concepts are filtered by tiers (generic → domain-specific → named entity → cross-table) to prevent noise from overly common concepts.

**Single-Call Generation**: UFD, SFD, and semantic profiles are generated in a single LLM call with structured output, reducing API costs by ~66% compared to separate calls.

## Evaluation

Evaluated on the NTCIR dataset discovery benchmark (46,615 datasets, 32 test queries) and Kaggle case studies. See the paper for detailed results.

## License

MIT License
