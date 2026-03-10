"""
Main experiment pipeline for SGDDG.

Orchestrates profiling, KG matching, routing, filtering,
reasoning, and unified metadata generation.
"""

import os
import time
import logging
import traceback
import concurrent.futures
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd

from .unified_generator import UnifiedMetadataGenerator
from .cache import ProfileCache, KGMatchCache
from .problem_classifier import ProblemClassifier
from .adaptive_router import AdaptiveRouter
from .tier0_filter import Tier0Filter
from .matchers import MultiGranularityMatcher, ContextAwareMatcher
from .conflict_resolver import SemanticStrengthLabeler
from .relationship_reasoner import RelationshipReasoner

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline output."""
    semantic_profiles: Dict[str, Any] = field(default_factory=dict)
    ufd: Dict[str, Any] = field(default_factory=dict)
    sfd: Dict[str, Any] = field(default_factory=dict)
    physical_profiles: list = field(default_factory=list)
    kg_matches: list = field(default_factory=list)
    dataset_context: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


def _process_column(
    col_name: str,
    df_column: pd.Series,
    profiler,
    matcher,
    p_cache: ProfileCache,
    k_cache: KGMatchCache,
    classifier: ProblemClassifier = None,
    adaptive_router: AdaptiveRouter = None,
    tier0_filter: Tier0Filter = None,
    strength_labeler: SemanticStrengthLabeler = None,
    multi_matcher: MultiGranularityMatcher = None,
    context_matcher: ContextAwareMatcher = None,
    dataset_name: str = None
) -> Tuple[str, Dict, Dict, bool, bool, Dict]:
    """Process a single column: profile, classify, route, match, filter, and label."""

    # Physical profiling with cache
    cached_profile = p_cache.get(df_column) if p_cache else None
    p_hit = cached_profile is not None

    if cached_profile:
        phy_profile = cached_profile
    else:
        try:
            phy_profile = profiler.profile(df_column, col_name)
            if p_cache:
                p_cache.save(df_column, phy_profile)
        except Exception as e:
            logger.error(f"Profiling failed for {col_name}: {e}")
            phy_profile = {"column_name": col_name, "error": str(e)}

    # Schema matching with classification and routing
    sample_values = phy_profile.get("sample_values", [])
    cached_match = k_cache.get(col_name, sample_values) if k_cache else None
    k_hit = False
    routing_info = {}

    # Classify column to determine if KG lookup is needed
    if classifier:
        classification = classifier.classify(col_name, phy_profile)
        routing_info["problem_type"] = classification.problem_type.value
        routing_info["match_strategy"] = classification.match_strategy.value
        routing_info["should_query_kg"] = classification.should_query_kg
        routing_info["classifier_reason"] = classification.reason

        if not classification.should_query_kg:
            match_result = {
                "status": "skipped",
                "reason": classification.reason,
                "problem_type": classification.problem_type.value
            }
            return (col_name, phy_profile, match_result, p_hit, False, routing_info)

    # Adaptive routing for confidence-based injection policy
    if adaptive_router:
        routing_decision = adaptive_router.route(phy_profile)
        routing_info["confidence_score"] = routing_decision.confidence_score
        routing_info["confidence_level"] = routing_decision.confidence_level.value
        routing_info["injection_policy"] = routing_decision.injection_policy.value
        routing_info["max_kg_concepts"] = routing_decision.max_kg_concepts
        routing_info["router_reason"] = routing_decision.reason

        if routing_decision.injection_policy.value == "SKIP":
            match_result = {
                "status": "skipped",
                "reason": f"Router SKIP policy: {routing_decision.reason}",
                "routing_decision": routing_decision.injection_policy.value
            }
            return (col_name, phy_profile, match_result, p_hit, False, routing_info)

    if cached_match:
        match_result = cached_match
        k_hit = True
    else:
        try:
            if context_matcher:
                match_result = context_matcher.match_column_with_context(
                    phy_profile, enable_inference=True
                )
            else:
                match_result = matcher.match_column(phy_profile)

            # Multi-granularity signal fusion
            if multi_matcher:
                problem_type = routing_info.get("problem_type", "default")
                fused_matches = multi_matcher.match_column(
                    col_name, sample_values, phy_profile,
                    dataset_name=dataset_name,
                    problem_type=problem_type,
                    kg_match=match_result
                )
                if fused_matches and match_result.get("status") == "matched":
                    match_result["score"] = fused_matches[0].final_score
                    match_result["multi_signal_reason"] = fused_matches[0].reason

            # Tier 0 filtering and semantic strength labeling
            if match_result.get("status") == "matched":
                concept_name = match_result.get("concept", {}).get("display_name", "")
                score = match_result.get("score", 0.0)

                if tier0_filter:
                    filter_res = tier0_filter.filter(concept_name, score)
                    match_result["tier"] = filter_res.tier.value
                    match_result["should_inject"] = filter_res.should_inject
                    match_result["filter_reason"] = filter_res.reason

                    if not filter_res.should_inject:
                        match_result["status"] = "filtered"
                        match_result["reason"] = f"Filtered by Tier 0: {filter_res.reason}"

                if strength_labeler and match_result["status"] == "matched":
                    labeled = strength_labeler.label(concept_name, score, phy_profile)
                    match_result["semantic_strength"] = labeled.semantic_strength.value
                    match_result["adjusted_score"] = labeled.adjusted_score
                    match_result["label_reason"] = labeled.label_reason

                    if labeled.semantic_strength.value == "UNCERTAIN":
                        match_result["status"] = "uncertain"
                        match_result["reason"] = labeled.label_reason

            if k_cache:
                k_cache.save(col_name, sample_values, match_result)
        except Exception as e:
            logger.error(f"Matching failed for {col_name}: {e}")
            match_result = {"status": "error", "error": str(e)}

    return (col_name, phy_profile, match_result, p_hit, k_hit, routing_info)


def run_pipeline(
    dataset_path: str,
    kg_data_dir: str = "data",
    dataset_name: str = "dataset",
    api_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    enable_parallel: bool = True,
    max_workers: int = 10,
    enable_cache: bool = True,
    use_kg_enhancement: bool = True,
) -> PipelineResult:
    """Run the full SGDDG pipeline on a dataset.

    Args:
        dataset_path: Path to the input CSV file.
        kg_data_dir: Directory containing KG data (schema_kg.db, vector_index.faiss).
        dataset_name: Human-readable name of the dataset.
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
        openai_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        enable_parallel: Run column processing in parallel.
        max_workers: Thread pool size for parallel processing.
        enable_cache: Enable profile and KG match caching.
        use_kg_enhancement: Enable knowledge graph concept injection.

    Returns:
        PipelineResult with generated metadata and statistics.
    """
    start_total = time.time()
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    openai_key = openai_key or os.environ.get("OPENAI_API_KEY")

    # Resolve KG vector index path
    expected_vector_path = os.path.join(kg_data_dir, "domain_vectors.faiss")
    actual_vector_path = os.path.join(kg_data_dir, "vector_index.faiss")
    if os.path.exists(actual_vector_path) and not os.path.exists(expected_vector_path):
        os.symlink("vector_index.faiss", expected_vector_path)

    # Initialize components
    logger.info("Initializing pipeline components...")

    try:
        from .profiling import ColumnProfiler, SchemaMatcher
        from .kg import KnowledgeGraphManager
    except ImportError:
        from profiling import ColumnProfiler, SchemaMatcher
        from kg import KnowledgeGraphManager

    kg_manager = KnowledgeGraphManager(kg_directory=kg_data_dir)
    col_profiler = ColumnProfiler()
    schema_matcher = SchemaMatcher(kg_manager)
    context_aware_matcher = ContextAwareMatcher(schema_matcher)

    unified_gen = UnifiedMetadataGenerator(
        api_key=api_key,
        use_optimized=True,
        optimized_config={
            'gemini_key': api_key,
            'openai_key': openai_key,
            'use_kg': use_kg_enhancement,
            'prefer_cheap': True,
            'enable_cache': enable_cache
        }
    )

    profile_cache = ProfileCache() if enable_cache else None
    kg_match_cache = KGMatchCache() if enable_cache else None
    problem_classifier = ProblemClassifier()
    adaptive_router = AdaptiveRouter()
    tier0_filter = Tier0Filter()
    strength_labeler = SemanticStrengthLabeler()
    multi_matcher = MultiGranularityMatcher()

    db_path = os.path.join(kg_data_dir, "kg", "domain_knowledge.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(kg_data_dir, "domain_knowledge.db")
    reasoner = RelationshipReasoner(db_path=db_path)

    # Load data
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

    # Phase 1: Column-level profiling and matching
    start_pm = time.time()
    results_map = {}

    task_args = dict(
        profiler=col_profiler, matcher=schema_matcher,
        p_cache=profile_cache, k_cache=kg_match_cache,
        classifier=problem_classifier, adaptive_router=adaptive_router,
        tier0_filter=tier0_filter, strength_labeler=strength_labeler,
        multi_matcher=multi_matcher, context_matcher=context_aware_matcher,
        dataset_name=dataset_name
    )

    if enable_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_column, col, df[col], **task_args): col
                for col in df.columns
            }
            for future in concurrent.futures.as_completed(futures):
                col_name = futures[future]
                try:
                    results_map[col_name] = future.result()
                except Exception as e:
                    logger.error(f"Column task failed for {col_name}: {e}")
    else:
        for col in df.columns:
            results_map[col] = _process_column(col, df[col], **task_args)

    # Collect results in original column order
    physical_profiles = []
    kg_matches = []
    cache_hits_p, cache_hits_k, skip_kg_count = 0, 0, 0

    for col in df.columns:
        if col not in results_map:
            continue
        _, phy, match, p_hit, k_hit, _ = results_map[col]
        physical_profiles.append(phy)
        kg_matches.append(match)
        cache_hits_p += int(p_hit)
        cache_hits_k += int(k_hit)
        if match.get("status") == "skipped":
            skip_kg_count += 1

    pm_duration = time.time() - start_pm
    logger.info(
        f"Phase 1 complete: {pm_duration:.2f}s, "
        f"cache hits: {cache_hits_p}p/{cache_hits_k}k, "
        f"skipped: {skip_kg_count}/{len(df.columns)}"
    )

    # Phase 2: Relational reasoning
    col_matches_map = {col: kg_matches[i] for i, col in enumerate(df.columns)}
    reasoning_res = reasoner.infer_dataset_context(col_matches_map)

    dataset_context = ""
    if reasoning_res.get("status") == "success":
        dataset_context = reasoning_res.get("dataset_insight", "")
        logger.info(f"Reasoning found {len(reasoning_res.get('internal_links', []))} relationships")

    # Phase 3: Unified metadata generation
    start_gen = time.time()

    try:
        all_metadata = unified_gen.generate_all_metadata(
            dataset_name=dataset_name,
            physical_profiles=physical_profiles,
            kg_matches=kg_matches,
            dataset_context=dataset_context
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        traceback.print_exc()
        return PipelineResult(
            physical_profiles=physical_profiles,
            kg_matches=kg_matches,
            stats={"error": str(e)}
        )

    gen_duration = time.time() - start_gen
    total_duration = time.time() - start_total

    result = PipelineResult(
        semantic_profiles=all_metadata.get("semantic_profiles", {}),
        ufd=all_metadata.get("ufd", {}),
        sfd=all_metadata.get("sfd", {}),
        physical_profiles=physical_profiles,
        kg_matches=kg_matches,
        dataset_context=dataset_context,
        stats={
            "total_time": total_duration,
            "profiling_time": pm_duration,
            "generation_time": gen_duration,
            "columns": len(df.columns),
            "cache_hits_profile": cache_hits_p,
            "cache_hits_kg": cache_hits_k,
            "kg_skipped": skip_kg_count,
        }
    )

    logger.info(
        f"Pipeline complete: {total_duration:.2f}s total "
        f"({pm_duration:.2f}s profiling, {gen_duration:.2f}s generation)"
    )

    return result
