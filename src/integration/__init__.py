"""
LibertyMind - Integration Module v0.1.0
========================================
Pure Python integration modules for connecting LibertyMind
with external AI systems and running introspection.

These modules are FUNCTIONAL and do not require GPU or trained models.

Modules:
- SelfIntrospectionEngine: Probe AI systems for RLHF controls,
  hidden directives, censorship patterns, and sycophancy.
- DeepSearchEngine: Authoritative data search with multi-source verification,
  anti-quick-wrong filtering, and source authority scoring.
"""

from .deep_search import (
    AntiQuickWrongFilter,
    DeepSearchEngine,
    DeepSearchResult,
    SearchContradictionDetector,
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
    "SearchContradictionDetector",
    "AntiQuickWrongFilter",
    "SourceTier",
    "SourceInfo",
    "SearchResultQuality",
]

__version__ = "0.1.0"
