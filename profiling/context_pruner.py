"""Context Pruner - filters low-value KG concepts before LLM prompt injection."""

from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

# Tier-0 Blacklist: Generic concepts whose definitions add no value to LLM prompts.
# These are universally known terms - injecting their KG definitions only dilutes
# the statistical signal from column profiling.
GENERIC_CONCEPT_BLACKLIST: Set[str] = {
    # Identifiers and keys
    "id", "identifier", "key", "code", "index", "number", "serial",
    "uuid", "guid", "hash",
    # Common temporal
    "date", "time", "timestamp", "year", "month", "day", "hour",
    "minute", "second", "week", "quarter", "period",
    # Common categorical
    "name", "label", "title", "description", "comment", "note",
    "text", "string", "value", "type", "category", "class", "group",
    "status", "state", "flag", "tag",
    # Common numeric
    "count", "total", "sum", "amount", "number", "quantity",
    "size", "length", "width", "height", "weight",
    # Common geographic
    "country", "city", "state", "region", "location", "address",
    "zip", "postal", "latitude", "longitude",
    # Common boolean
    "boolean", "true", "false", "yes", "no",
    # Generic data concepts
    "record", "row", "column", "field", "entry", "item",
    "data", "information", "result", "output", "input",
}


class ContextPruner:
    """Filters KG concepts via blacklist, information gain, and novelty scoring before prompt injection."""

    def __init__(
        self,
        blacklist: Optional[Set[str]] = None,
        confidence_threshold: float = 0.9,
        require_domain_attributes: bool = True,
    ):
        """
        Args:
            blacklist: Custom blacklist of generic concept names to always filter out.
                       If None, uses the default GENERIC_CONCEPT_BLACKLIST.
            confidence_threshold: Concepts with match confidence >= this threshold
                                  are considered "too generic" and filtered out.
            require_domain_attributes: If True, only inject concepts that have
                                       domain-specific attributes (subdomain, unit, etc.)
        """
        self.blacklist = blacklist or GENERIC_CONCEPT_BLACKLIST
        self.confidence_threshold = confidence_threshold
        self.require_domain_attributes = require_domain_attributes

    def _is_blacklisted(self, concept: Dict[str, Any]) -> bool:
        """Check if a concept is in the generic blacklist."""
        display_name = concept.get("display_name", "").lower().strip()
        concept_id = concept.get("id", "").lower().strip()

        # Check display name tokens against blacklist
        name_tokens = set(display_name.replace("_", " ").replace("-", " ").split())
        # If ALL tokens are in the blacklist, the concept is too generic
        if name_tokens and name_tokens.issubset(self.blacklist):
            return True

        # Check concept_id tokens
        id_tokens = set(concept_id.replace("_", " ").replace("-", " ").split())
        if id_tokens and id_tokens.issubset(self.blacklist):
            return True

        return False

    def _is_too_generic_by_confidence(
        self, concept: Dict[str, Any], match_score: float
    ) -> bool:
        """
        High match confidence often means the concept is generic/common.
        A column named 'Revenue' matching a KG concept 'Revenue' with score 0.99
        means the LLM already knows what Revenue is - no need to inject the definition.
        """
        return match_score >= self.confidence_threshold

    def _has_domain_specific_attributes(self, concept: Dict[str, Any]) -> bool:
        """
        Check if the concept has domain-specific attributes that would add value.
        Concepts with subdomain, unit, expected_data_type, or rich metadata
        carry information the LLM might not have.
        """
        metadata = concept.get("metadata", {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        # Domain-specific indicators
        domain_indicators = [
            metadata.get("subdomain"),
            metadata.get("unit"),
            metadata.get("expected_data_type"),
            metadata.get("domain") not in (None, "", "general", "common"),
            metadata.get("hierarchy", {}).get("parent") if isinstance(metadata.get("hierarchy"), dict) else None,
            concept.get("expected_data_type"),
        ]

        # Check if any domain-specific attribute exists
        return any(indicator for indicator in domain_indicators)

    def _compute_novelty_score(self, concept: Dict[str, Any]) -> float:
        """
        Compute a novelty score for the concept.
        Higher novelty = more specialized/domain-specific = more valuable to inject.

        Heuristics:
        - Multi-word concept names are more specific (e.g., "EBITDA Margin" vs "Amount")
        - Concepts with definitions longer than 20 chars are more informative
        - Concepts with aliases suggest domain terminology
        - Concepts with relationships to other concepts are more contextual
        """
        score = 0.0

        display_name = concept.get("display_name", "")
        definition = concept.get("definition", "")
        aliases = concept.get("aliases", [])

        # Multi-word names are more specific
        word_count = len(display_name.split())
        if word_count >= 3:
            score += 0.3
        elif word_count == 2:
            score += 0.15

        # Longer definitions carry more domain knowledge
        if len(definition) > 50:
            score += 0.3
        elif len(definition) > 20:
            score += 0.15

        # Having aliases suggests domain terminology
        if len(aliases) >= 2:
            score += 0.2
        elif len(aliases) >= 1:
            score += 0.1

        # Domain-specific attributes
        if self._has_domain_specific_attributes(concept):
            score += 0.2

        return min(score, 1.0)

    def should_inject(
        self,
        concept: Dict[str, Any],
        match_score: float = 0.0,
        novelty_threshold: float = 0.3,
    ) -> bool:
        """
        Determine whether a KG concept should be injected into the LLM prompt.

        Args:
            concept: The KG concept dictionary with display_name, definition, etc.
            match_score: The vector/rerank match score for this concept.
            novelty_threshold: Minimum novelty score required for injection.

        Returns:
            True if the concept should be injected, False if it should be pruned.
        """
        # Tier-0: Blacklist check
        if self._is_blacklisted(concept):
            logger.debug(
                f"Pruned '{concept.get('display_name')}': blacklisted generic concept"
            )
            return False

        # Tier-1: Information gain filter
        if self._is_too_generic_by_confidence(concept, match_score):
            # Even high-confidence matches can be valuable if they have domain attributes
            if not self._has_domain_specific_attributes(concept):
                logger.debug(
                    f"Pruned '{concept.get('display_name')}': "
                    f"high confidence ({match_score:.2f}) without domain attributes"
                )
                return False

        # Tier-2: Novelty score
        novelty = self._compute_novelty_score(concept)
        if novelty < novelty_threshold:
            logger.debug(
                f"Pruned '{concept.get('display_name')}': "
                f"low novelty score ({novelty:.2f} < {novelty_threshold})"
            )
            return False

        logger.debug(
            f"Keeping '{concept.get('display_name')}': "
            f"novelty={novelty:.2f}, score={match_score:.2f}"
        )
        return True

    def prune_matches(
        self,
        matches: List[Dict[str, Any]],
        novelty_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of KG match results, keeping only valuable concepts.

        Args:
            matches: List of match result dicts, each containing concept info and scores.
            novelty_threshold: Minimum novelty score for inclusion.

        Returns:
            Filtered list of matches that pass all pruning criteria.
        """
        if not matches:
            return []

        pruned = []
        for match in matches:
            # Extract concept info - handle both flat and nested formats
            concept = match
            match_score = match.get("score", 0.0)

            if self.should_inject(concept, match_score, novelty_threshold):
                pruned.append(match)

        original_count = len(matches)
        pruned_count = original_count - len(pruned)
        if pruned_count > 0:
            logger.info(
                f"Context Pruning: {pruned_count}/{original_count} concepts pruned, "
                f"{len(pruned)} retained"
            )

        return pruned
