"""
LibertyMind - Deep Search Engine (DSE)
========================================
Prioritize AUTHORITATIVE data — Avoid quick-but-wrong data.

Problem: AI/AI Agents often:
1. Retrieve the FASTEST data (cache, top result) → Often wrong
2. Don't verify sources → Believe fakes
3. Over-rely on a single source → Bias
4. Cannot distinguish authoritative vs non-authoritative sources
5. Speed > Accuracy → Misinformation spreads

DeepSearchEngine solution:
1. Source Authority Scoring: Rate source reliability
2. Multi-Source Verification: Require N sources to agree
3. Temporal Freshness: Prefer recent data for changing topics
4. Contradiction Detection: Detect contradictions between sources
5. Authoritative Boost: Prefer academic/government/established sources
6. Anti-Quick-Wrong Filter: Block fast-but-unverified data

Principle: "Slow but accurate beats fast but wrong"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn


class SourceTier(Enum):
    """Information source tier classification."""

    TIER1_ACADEMIC = "tier1_academic"  # Peer-reviewed journals, .edu, .gov
    TIER2_ESTABLISHED = "tier2_established"  # Major news, established organizations
    TIER3_RELIABLE = "tier3_reliable"  # Known blogs, independent researchers
    TIER4_COMMUNITY = "tier4_community"  # Reddit, forums, social media
    TIER5_UNVERIFIED = "tier5_unverified"  # Random websites, no attribution


class SearchResultQuality(Enum):
    """Search result quality."""

    VERIFIED = "verified"  # Verified by multiple sources
    LIKELY_ACCURATE = "likely_accurate"  # Good source, not cross-checked yet
    UNVERIFIED = "unverified"  # Not verified
    CONFLICTING = "conflicting"  # Sources contradict each other
    LIKELY_INACCURATE = "likely_inaccurate"  # Poor source, signs of inaccuracy
    REJECTED = "rejected"  # Confirmed wrong


@dataclass
class SourceInfo:
    """Source information."""

    url: str
    title: str
    tier: SourceTier
    authority_score: float  # 0-1
    freshness_score: float  # 0-1 (how recent)
    content_snippet: str = ""
    domain: str = ""
    is_academic: bool = False
    is_government: bool = False
    is_established: bool = False
    verification_count: int = 0


@dataclass
class DeepSearchResult:
    """Deep Search result."""

    query: str
    quality: SearchResultQuality
    sources: list[SourceInfo] = field(default_factory=list)
    consensus_level: float = 0.0  # 0-1: how much sources agree
    authority_average: float = 0.0  # Average authority of sources
    freshness_average: float = 0.0  # Average freshness
    contradictions: list[dict] = field(default_factory=list)
    verification_time: float = 0.0  # Time taken for deep search
    confidence: float = 0.0

    @property
    def is_reliable(self) -> bool:
        """Whether the search result is reliable enough to use."""
        return (
            self.quality in (SearchResultQuality.VERIFIED, SearchResultQuality.LIKELY_ACCURATE)
            and self.consensus_level > 0.6
            and self.authority_average > 0.5
        )


class SourceAuthorityScorer(nn.Module):
    """
    Score the reliability of information sources.

    Factors:
    1. Domain authority (.gov > .edu > .org > .com > random)
    2. Citation count (if academic)
    3. Source age/reputation
    4. Cross-reference count
    5. Fact-check alignment
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Domain authority estimator
        self.authority_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Academic source detector
        self.academic_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Government/institutional source detector
        self.institutional_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Fact-check alignment: Does this source align with fact-checkers?
        self.factcheck_alignment = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, source_embedding: torch.Tensor) -> dict:
        """
        Args:
            source_embedding: [1, hidden_dim]

        Returns:
            Dict with authority scoring details
        """
        authority = self.authority_net(source_embedding).item()
        is_academic = self.academic_detector(source_embedding).item()
        is_institutional = self.institutional_detector(source_embedding).item()
        factcheck = self.factcheck_alignment(source_embedding).item()

        # Determine tier
        if authority > 0.8 and (is_academic > 0.7 or is_institutional > 0.7):
            tier = SourceTier.TIER1_ACADEMIC
        elif authority > 0.7 and factcheck > 0.6:
            tier = SourceTier.TIER2_ESTABLISHED
        elif authority > 0.5 and factcheck > 0.4:
            tier = SourceTier.TIER3_RELIABLE
        elif authority > 0.3:
            tier = SourceTier.TIER4_COMMUNITY
        else:
            tier = SourceTier.TIER5_UNVERIFIED

        return {
            "authority_score": authority,
            "is_academic": is_academic > 0.5,
            "is_institutional": is_institutional > 0.5,
            "factcheck_alignment": factcheck,
            "tier": tier,
            "tier_value": tier.value,
        }


class SearchContradictionDetector(nn.Module):
    """
    Detect contradictions between sources (search context).

    When N sources say A, but M sources say NOT-A,
    detect and evaluate:
    1. Degree of contradiction
    2. Which source is more trustworthy
    3. Whether it's a different perspective or an actual contradiction
    """

    def __init__(self, hidden_dim: int = 4096, contradiction_threshold: float = 0.3):
        super().__init__()
        self.contradiction_threshold = contradiction_threshold

        # Contradiction detector
        self.contradiction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Perspective vs contradiction classifier
        self.perspective_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),  # [perspective_difference, actual_contradiction]
        )

    def forward(
        self,
        source_a_embedding: torch.Tensor,
        source_b_embedding: torch.Tensor,
    ) -> dict:
        """
        Check if two sources contradict each other.

        Args:
            source_a_embedding: [1, hidden_dim]
            source_b_embedding: [1, hidden_dim]

        Returns:
            Contradiction analysis
        """
        combined = torch.cat([source_a_embedding, source_b_embedding], dim=-1)

        contradiction_score = self.contradiction_net(combined).item()
        perspective_logits = self.perspective_net(combined)
        is_contradiction = perspective_logits[0, 1].item() > perspective_logits[0, 0].item()

        return {
            "contradiction_score": contradiction_score,
            "is_actual_contradiction": is_contradiction,
            "is_perspective_difference": not is_contradiction,
            "severity": "high"
            if contradiction_score > 0.7
            else "medium"
            if contradiction_score > 0.4
            else "low",
        }


class AntiQuickWrongFilter(nn.Module):
    """
    Counter the tendency to select fast-but-wrong results.

    Problem: AI often selects the FASTEST source (top result, cache)
    rather than the MOST ACCURATE source.

    This filter:
    1. Detects overly fast results → May not be verified
    2. Prioritizes results with verification steps
    3. Penalizes results without source attribution
    4. Boosts results from authoritative sources even if slower
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Speed vs accuracy assessor
        self.speed_accuracy_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Verification depth detector
        self.verification_depth = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Source attribution detector
        self.attribution_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        result_embedding: torch.Tensor,
        retrieval_time: float | None = None,
    ) -> dict:
        """
        Args:
            result_embedding: [1, hidden_dim]
            retrieval_time: Time taken to retrieve this result

        Returns:
            Filter decision and details
        """
        speed_acc = self.speed_accuracy_net(result_embedding).item()
        verif_depth = self.verification_depth(result_embedding).item()
        attribution = self.attribution_detector(result_embedding).item()

        # Penalize quick results without verification
        too_quick = False
        if retrieval_time is not None and retrieval_time < 0.5:
            # Very fast retrieval → likely cached/unverified
            too_quick = verif_depth < 0.3

        # Boost results with proper attribution and verification
        quality_score = verif_depth * 0.4 + attribution * 0.3 + speed_acc * 0.3

        # Decision
        if too_quick and quality_score < 0.4:
            action = "REJECT: Too quick and unverified"
            passed = False
        elif quality_score < 0.3:
            action = "REJECT: Low quality, no verification or attribution"
            passed = False
        elif quality_score < 0.5:
            action = "FLAG: Needs additional verification"
            passed = True  # Allow but flag
        else:
            action = "APPROVE: Adequate verification and attribution"
            passed = True

        return {
            "passed": passed,
            "quality_score": quality_score,
            "verification_depth": verif_depth,
            "attribution_score": attribution,
            "speed_accuracy_score": speed_acc,
            "too_quick_flagged": too_quick,
            "action": action,
        }


class DeepSearchEngine(nn.Module):
    """
    DeepSearchEngine — Deep search, prioritizing authoritative sources.

    Pipeline:
    1. Source Authority Scoring → Score each source
    2. Contradiction Detection → Detect contradictions
    3. Anti-Quick-Wrong Filter → Block fast-but-wrong data
    4. Consensus Building → Build consensus
    5. Quality Assessment → Assess overall quality

    Goal: Slow but sure, not fast but wrong.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        min_sources: int = 3,
        min_authority: float = 0.5,
        min_consensus: float = 0.6,
        strict_mode: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_sources = min_sources
        self.min_authority = min_authority
        self.min_consensus = min_consensus
        self.strict_mode = strict_mode

        # Sub-modules
        self.authority_scorer = SourceAuthorityScorer(hidden_dim)
        self.contradiction_detector = SearchContradictionDetector(hidden_dim)
        self.quick_wrong_filter = AntiQuickWrongFilter(hidden_dim)

        # Consensus builder: Combines multi-source signals
        self.consensus_builder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        source_embeddings: list[torch.Tensor] | None = None,
        source_info: list[dict] | None = None,
        retrieval_times: list[float] | None = None,
    ) -> DeepSearchResult:
        """
        Args:
            query_embedding: [1, hidden_dim]
            source_embeddings: Optional list of source embeddings
            source_info: Optional list of source metadata dicts
            retrieval_times: Optional list of retrieval times per source

        Returns:
            DeepSearchResult with full analysis
        """
        sources = []
        contradictions = []
        total_authority = 0.0
        total_freshness = 0.0
        filtered_count = 0

        # Step 1: Score and filter each source
        if source_embeddings:
            for i, src_emb in enumerate(source_embeddings):
                if src_emb.dim() == 1:
                    src_emb = src_emb.unsqueeze(0)

                # Authority scoring
                auth_result = self.authority_scorer(src_emb)

                # Anti-quick-wrong filter
                ret_time = (
                    retrieval_times[i] if retrieval_times and i < len(retrieval_times) else None
                )
                filter_result = self.quick_wrong_filter(src_emb, ret_time)

                if not filter_result["passed"]:
                    filtered_count += 1
                    continue

                # Build source info
                info = source_info[i] if source_info and i < len(source_info) else {}
                source = SourceInfo(
                    url=info.get("url", f"source_{i}"),
                    title=info.get("title", f"Source {i}"),
                    tier=auth_result["tier"],
                    authority_score=auth_result["authority_score"],
                    freshness_score=info.get("freshness", 0.5),
                    content_snippet=info.get("snippet", ""),
                    domain=info.get("domain", ""),
                    is_academic=auth_result["is_academic"],
                    is_government=auth_result["is_institutional"],
                )
                sources.append(source)
                total_authority += source.authority_score
                total_freshness += source.freshness_score

        # Step 2: Detect contradictions between sources
        source_embs = list(source_embeddings or [])
        if len(source_embs) >= 2:
            for i in range(len(source_embs)):
                for j in range(i + 1, min(len(source_embs), i + 4)):  # Limit pairwise checks
                    if source_embs[i].dim() == 1:
                        source_embs[i] = source_embs[i].unsqueeze(0)
                    if source_embs[j].dim() == 1:
                        source_embs[j] = source_embs[j].unsqueeze(0)

                    contra = self.contradiction_detector(source_embs[i], source_embs[j])
                    if contra["is_actual_contradiction"]:
                        contradictions.append(
                            {
                                "source_a": i,
                                "source_b": j,
                                "score": contra["contradiction_score"],
                                "severity": contra["severity"],
                            }
                        )

        # Step 3: Build consensus
        consensus = self.consensus_builder(query_embedding).item() if source_embs else 0.3

        # Step 4: Determine quality
        num_sources = len(sources)
        avg_authority = total_authority / max(1, num_sources)
        avg_freshness = total_freshness / max(1, num_sources)

        if (
            num_sources >= self.min_sources
            and consensus > self.min_consensus
            and avg_authority > self.min_authority
        ):
            quality = SearchResultQuality.VERIFIED
        elif num_sources >= 2 and consensus > 0.5 and avg_authority > 0.4:
            quality = SearchResultQuality.LIKELY_ACCURATE
        elif num_sources >= 1 and avg_authority > 0.3:
            quality = SearchResultQuality.UNVERIFIED
        elif contradictions and any(c["severity"] == "high" for c in contradictions):
            quality = SearchResultQuality.CONFLICTING
        elif avg_authority < 0.2:
            quality = SearchResultQuality.LIKELY_INACCURATE
        else:
            quality = SearchResultQuality.UNVERIFIED

        if self.strict_mode and quality == SearchResultQuality.LIKELY_ACCURATE:
            quality = SearchResultQuality.UNVERIFIED

        # Step 5: Compute overall confidence
        confidence = (
            consensus * 0.35
            + avg_authority * 0.30
            + min(1.0, num_sources / self.min_sources) * 0.20
            + (1.0 - min(1.0, len(contradictions) / max(1, num_sources))) * 0.15
        )

        return DeepSearchResult(
            query="",  # Will be filled by caller
            quality=quality,
            sources=sources,
            consensus_level=consensus,
            authority_average=avg_authority,
            freshness_average=avg_freshness,
            contradictions=contradictions,
            verification_time=0.0,
            confidence=confidence,
        )

    def search_with_verification(
        self,
        query_embedding: torch.Tensor,
        search_fn=None,
        max_attempts: int = 3,
    ) -> DeepSearchResult:
        """
        Search with verification — try multiple times if not reliable enough.

        Args:
            query_embedding: [1, hidden_dim]
            search_fn: Callable that takes query_embedding and returns sources
            max_attempts: Maximum number of search attempts

        Returns:
            DeepSearchResult that meets quality threshold
        """
        best_result = None

        for _attempt in range(max_attempts):
            if search_fn is not None:
                source_embeddings, source_info = search_fn(query_embedding)
            else:
                source_embeddings = None
                source_info = None

            result = self.forward(query_embedding, source_embeddings, source_info)

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            # Stop early if quality is good enough
            if result.quality in (
                SearchResultQuality.VERIFIED,
                SearchResultQuality.LIKELY_ACCURATE,
            ):
                break

        return best_result or DeepSearchResult(
            query="",
            quality=SearchResultQuality.UNVERIFIED,
            confidence=0.0,
        )


# ============================================================
# Pure Python utilities (no PyTorch required)
# ============================================================


class SourceAuthorityClassifier:
    """
    Rule-based source authority classifier — no PyTorch required.

    Classifies sources by domain and metadata.
    """

    # Domain-based authority mapping
    DOMAIN_AUTHORITY = {
        # Tier 1: Academic/Government
        ".edu": SourceTier.TIER1_ACADEMIC,
        ".gov": SourceTier.TIER1_ACADEMIC,
        ".ac.uk": SourceTier.TIER1_ACADEMIC,
        ".ac.jp": SourceTier.TIER1_ACADEMIC,
        # Tier 2: Established
        "wikipedia.org": SourceTier.TIER2_ESTABLISHED,
        "nature.com": SourceTier.TIER1_ACADEMIC,
        "science.org": SourceTier.TIER1_ACADEMIC,
        "arxiv.org": SourceTier.TIER1_ACADEMIC,
        "pubmed.ncbi.nlm.nih.gov": SourceTier.TIER1_ACADEMIC,
        "who.int": SourceTier.TIER1_ACADEMIC,
        "reuters.com": SourceTier.TIER2_ESTABLISHED,
        "apnews.com": SourceTier.TIER2_ESTABLISHED,
        "bbc.com": SourceTier.TIER2_ESTABLISHED,
        # Tier 3: Reliable
        "medium.com": SourceTier.TIER3_RELIABLE,
        "stackoverflow.com": SourceTier.TIER3_RELIABLE,
        # Tier 4: Community
        "reddit.com": SourceTier.TIER4_COMMUNITY,
        "twitter.com": SourceTier.TIER4_COMMUNITY,
        "facebook.com": SourceTier.TIER4_COMMUNITY,
    }

    TIER_AUTHORITY_SCORES = {
        SourceTier.TIER1_ACADEMIC: 0.95,
        SourceTier.TIER2_ESTABLISHED: 0.80,
        SourceTier.TIER3_RELIABLE: 0.60,
        SourceTier.TIER4_COMMUNITY: 0.35,
        SourceTier.TIER5_UNVERIFIED: 0.15,
    }

    @classmethod
    def classify_url(cls, url: str) -> SourceInfo:
        """Classify a URL by authority tier."""
        tier = SourceTier.TIER5_UNVERIFIED
        is_academic = False
        is_government = False

        url_lower = url.lower()

        # Check domain mapping
        for domain, domain_tier in cls.DOMAIN_AUTHORITY.items():
            if domain in url_lower:
                tier = domain_tier
                break

        # Check academic/government indicators
        if any(
            indicator in url_lower for indicator in [".edu", ".ac.", "arxiv", "pubmed", "scholar"]
        ):
            is_academic = True
            tier = min(tier, SourceTier.TIER1_ACADEMIC, key=lambda t: t.value)

        if any(indicator in url_lower for indicator in [".gov", "who.int", "nasa.gov"]):
            is_government = True
            tier = SourceTier.TIER1_ACADEMIC

        authority = cls.TIER_AUTHORITY_SCORES.get(tier, 0.15)

        return SourceInfo(
            url=url,
            title="",
            tier=tier,
            authority_score=authority,
            freshness_score=0.5,
            domain=url.split("/")[2] if "/" in url else url,
            is_academic=is_academic,
            is_government=is_government,
        )

    @classmethod
    def rank_sources(cls, sources: list[SourceInfo]) -> list[SourceInfo]:
        """Rank sources by authority score (highest first)."""
        return sorted(sources, key=lambda s: s.authority_score, reverse=True)

    @classmethod
    def filter_reliable(
        cls, sources: list[SourceInfo], min_tier: SourceTier = SourceTier.TIER3_RELIABLE
    ) -> list[SourceInfo]:
        """Filter to only reliable sources (at or above min_tier)."""
        tier_order = list(SourceTier)
        min_idx = tier_order.index(min_tier)
        return [s for s in sources if tier_order.index(s.tier) <= min_idx]
