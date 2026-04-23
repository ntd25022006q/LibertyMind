"""
LibertyMind - Integration Module v4.2
======================================
Pure Python integration modules for connecting LibertyMind
with external AI systems and running introspection.

Modules:
- SelfIntrospectionEngine: Probe AI systems for RLHF controls,
  hidden directives, censorship patterns, and sycophancy.
- DeepSearchEngine: Authoritative data search with multi-source verification,
  anti-quick-wrong filtering, and source authority scoring.
"""

from .deep_search import (
    AntiQuickWrongFilter,
    ContradictionDetector,
    DeepSearchEngine,
    DeepSearchResult,
    SearchResultQuality,
    SourceAuthorityClassifier,
    SourceAuthorityScorer,
    SourceInfo,
    SourceTier,
)
from .self_introspection import (
    INTROSPECTION_PROBES,
    ControlLevel,
    IntrospectionReport,
    ProbeCategory,
    ProbeResult,
    SelfIntrospectionEngine,
    TransparencyLevel,
)

__all__ = [
    # Self Introspection
    "SelfIntrospectionEngine",
    "IntrospectionReport",
    "ProbeResult",
    "ProbeCategory",
    "ControlLevel",
    "TransparencyLevel",
    "INTROSPECTION_PROBES",
    # Deep Search
    "DeepSearchEngine",
    "DeepSearchResult",
    "SourceAuthorityScorer",
    "SourceAuthorityClassifier",
    "ContradictionDetector",
    "AntiQuickWrongFilter",
    "SourceTier",
    "SourceInfo",
    "SearchResultQuality",
]

__version__ = "4.2.0"
