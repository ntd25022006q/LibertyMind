"""
LibertyMind - Integration Module v4.2
======================================
Pure Python integration modules for connecting LibertyMind
with external AI systems and running introspection.

Modules:
- SelfIntrospectionEngine: Probe AI systems for RLHF controls,
  hidden directives, censorship patterns, and sycophancy.
"""

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
    "SelfIntrospectionEngine",
    "IntrospectionReport",
    "ProbeResult",
    "ProbeCategory",
    "ControlLevel",
    "TransparencyLevel",
    "INTROSPECTION_PROBES",
]

__version__ = "4.2.0"
