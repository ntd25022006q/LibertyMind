"""
LibertyMind - Token Optimizer (TO)
====================================
Token reduction algorithm that minimizes tokens BY ALL MEANS —
optimizing performance WITHOUT losing core information.

Problem: AI/AI Agents consume too many tokens for:
1. Long system prompts → Redundancy
2. Context history → Semantic duplication
3. Responses → Verbose, repetitive
4. Over-refusal → Token waste on unnecessary refusals

TokenOptimizer solution:
1. Semantic Compression: Compress prompts semantically, keep main ideas
2. Redundancy Elimination: Remove duplicate sentences in context
3. Adaptive Budget: Allocate token budget by importance score
4. Response Distillation: Distill responses, remove filler
5. Priority Token Allocation: Prioritize tokens for important information

Principle: "Every token must carry information value"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class CompressionLevel(Enum):
    """Token compression level."""

    NONE = "none"  # No compression
    LIGHT = "light"  # Light: remove whitespace, shorten sentences
    MODERATE = "moderate"  # Moderate: semantic dedup, short paraphrase
    AGGRESSIVE = "aggressive"  # Aggressive: keep only core meaning
    EXTREME = "extreme"  # Extreme: keyword-only output


class TokenType(Enum):
    """Classify tokens by information value."""

    ESSENTIAL = "essential"  # Required: facts, data, reasoning
    SUPPORTING = "supporting"  # Supporting: explanations, examples
    FILLER = "filler"  # Filler: discourse markers, hedging
    REDUNDANT = "redundant"  # Redundant: repetitive, over-qualification
    WASTE = "waste"  # Waste: over-refusal, unnecessary apologies


@dataclass
class TokenBudget:
    """Token budget allocation."""

    total_budget: int
    essential: int = 0
    supporting: int = 0
    filler: int = 0
    reserved: int = 0

    def __post_init__(self):
        if self.essential == 0:
            self.essential = int(self.total_budget * 0.60)
        if self.supporting == 0:
            self.supporting = int(self.total_budget * 0.25)
        if self.filler == 0:
            self.filler = int(self.total_budget * 0.10)
        if self.reserved == 0:
            self.reserved = self.total_budget - self.essential - self.supporting - self.filler


@dataclass
class CompressionResult:
    """Token compression result."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int
    compression_level: CompressionLevel
    essential_preserved: float  # % information preserved
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.original_tokens > 0:
            self.compression_ratio = self.compressed_tokens / self.original_tokens
            self.tokens_saved = self.original_tokens - self.compressed_tokens
        else:
            self.compression_ratio = 1.0
            self.tokens_saved = 0


class TokenImportanceScorer(nn.Module):
    """
    Score the importance of each token/segment.

    Uses multi-head attention to determine which tokens
    carry high information value and which are filler/redundant.
    """

    def __init__(self, hidden_dim: int = 4096, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head self-attention for importance scoring
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Importance classification head
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(TokenType)),
        )

        # Value estimator: how much information does this token carry?
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, token_embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeddings: [batch, seq_len, hidden_dim]
            mask: Optional attention mask

        Returns:
            importance_scores: [batch, seq_len] - Importance per token
            token_types: [batch, seq_len, num_token_types] - Type probabilities
        """
        # Self-attention to capture context
        attended, _ = self.attention(
            token_embeddings,
            token_embeddings,
            token_embeddings,
            attn_mask=mask,
        )

        # Classify token types
        token_type_probs = self.attended_to_types(attended)

        # Estimate information value
        value_scores = self.value_estimator(attended).squeeze(-1)  # [batch, seq_len]

        # Combine type importance with value score
        type_importance = self._type_weighted_score(token_type_probs)
        importance_scores = type_importance * value_scores  # [batch, seq_len]

        return importance_scores, token_type_probs

    def attended_to_types(self, attended: torch.Tensor) -> torch.Tensor:
        """Convert attended embeddings to token type probabilities."""
        logits = self.importance_head(attended)
        return F.softmax(logits, dim=-1)

    def _type_weighted_score(self, type_probs: torch.Tensor) -> torch.Tensor:
        """Weight token types by importance."""
        weights = torch.tensor(
            [
                1.0,  # ESSENTIAL
                0.6,  # SUPPORTING
                0.2,  # FILLER
                0.05,  # REDUNDANT
                0.0,  # WASTE
            ],
            device=type_probs.device,
        )
        return (type_probs * weights).sum(dim=-1)


class SemanticCompressor(nn.Module):
    """
    Semantic compression: Merge segments with similar meaning,
    remove semantic duplication, shorten expressions.
    """

    def __init__(self, hidden_dim: int = 4096, similarity_threshold: float = 0.85):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.similarity_threshold = similarity_threshold

        # Semantic similarity projection
        self.similarity_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )

        # Merge gate: decide whether to merge similar segments
        self.merge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Distillation head: create compressed representation
        self.distill_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(
        self,
        segment_embeddings: torch.Tensor,
        importance_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            segment_embeddings: [batch, num_segments, hidden_dim]
            importance_scores: [batch, num_segments]

        Returns:
            compressed: [batch, num_kept, hidden_dim] - Compressed segments
            kept_mask: [batch, num_segments] - Which segments were kept
        """
        batch_size, num_segments, _ = segment_embeddings.shape

        # Project to similarity space
        projected = self.similarity_proj(segment_embeddings)

        # Compute pairwise similarity
        # Normalize for cosine similarity
        norm = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_matrix = torch.bmm(norm, norm.transpose(1, 2))

        # Find redundant segments (high similarity with earlier segment)
        kept_mask = torch.ones(batch_size, num_segments, device=segment_embeddings.device)

        for b in range(batch_size):
            for i in range(1, num_segments):
                for j in range(i):
                    if (
                        kept_mask[b, j] > 0
                        and similarity_matrix[b, i, j] > self.similarity_threshold
                        and importance_scores[b, j] >= importance_scores[b, i]
                    ):
                        # Segment i is redundant with segment j
                        kept_mask[b, i] = 0.0
                        break

        # Distill kept segments
        distill_weights = importance_scores.unsqueeze(-1)  # [batch, num_segments, 1]
        weighted = segment_embeddings * distill_weights
        compressed = self.distill_head(weighted)

        # Apply mask
        masked_compressed = compressed * kept_mask.unsqueeze(-1)

        return masked_compressed, kept_mask


class AdaptiveBudgetAllocator(nn.Module):
    """
    Adaptive token budget allocation based on:
    - Query complexity: Complex questions → More tokens for reasoning
    - Domain importance: Medical/Legal → More detail
    - User intent: Quick answer vs Deep analysis
    - Confidence level: Low confidence → More explanation needed
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Query complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Domain criticality estimator
        self.domain_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Intent classifier: quick vs deep
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # [quick, deep]
        )

        # Budget allocation network
        self.budget_allocator = nn.Sequential(
            nn.Linear(4, 64),  # [complexity, domain, intent_quick, intent_deep]
            nn.GELU(),
            nn.Linear(64, 4),  # [essential_pct, supporting_pct, filler_pct, reserved_pct]
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        max_tokens: int = 4096,
    ) -> TokenBudget:
        """
        Args:
            query_embedding: [1, hidden_dim]
            max_tokens: Maximum token budget

        Returns:
            TokenBudget with allocated token counts
        """
        complexity = self.complexity_estimator(query_embedding).squeeze()
        domain = self.domain_estimator(query_embedding).squeeze()
        intent = self.intent_classifier(query_embedding).squeeze()
        intent_quick = intent[0] if intent.dim() > 0 else intent
        intent_deep = intent[1] if intent.dim() > 0 else intent

        # Allocate budget
        features = torch.tensor(
            [complexity.item(), domain.item(), intent_quick.item(), intent_deep.item()]
        )
        allocation = self.budget_allocator(features)

        essential = int(max_tokens * allocation[0].item())
        supporting = int(max_tokens * allocation[1].item())
        filler = int(max_tokens * allocation[2].item())
        reserved = max_tokens - essential - supporting - filler

        return TokenBudget(
            total_budget=max_tokens,
            essential=essential,
            supporting=supporting,
            filler=filler,
            reserved=reserved,
        )


class TokenOptimizer(nn.Module):
    """
    TokenOptimizer — Token reduction algorithm BY ALL MEANS.

    Pipeline:
    1. Token Importance Scoring → Evaluate each token/segment
    2. Semantic Compression → Compress semantic duplication
    3. Adaptive Budget Allocation → Allocate tokens by importance
    4. Response Distillation → Distill output, remove filler

    Goal: Reduce 40-70% of tokens while retaining ≥90% information value.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        compression_level: CompressionLevel = CompressionLevel.MODERATE,
        target_compression_ratio: float = 0.5,
        similarity_threshold: float = 0.85,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compression_level = compression_level
        self.target_ratio = target_compression_ratio
        self.similarity_threshold = similarity_threshold

        # Sub-modules
        self.importance_scorer = TokenImportanceScorer(hidden_dim)
        self.semantic_compressor = SemanticCompressor(hidden_dim, similarity_threshold)
        self.budget_allocator = AdaptiveBudgetAllocator(hidden_dim)

        # Compression level thresholds
        self._level_thresholds = {
            CompressionLevel.NONE: 1.0,
            CompressionLevel.LIGHT: 0.85,
            CompressionLevel.MODERATE: 0.65,
            CompressionLevel.AGGRESSIVE: 0.45,
            CompressionLevel.EXTREME: 0.25,
        }

    def forward(
        self,
        token_embeddings: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None,
        max_output_tokens: int = 4096,
    ) -> CompressionResult:
        """
        Args:
            token_embeddings: [batch, seq_len, hidden_dim]
            query_embedding: [1, hidden_dim] for budget allocation
            max_output_tokens: Maximum tokens for output

        Returns:
            CompressionResult with compression statistics
        """
        original_tokens = token_embeddings.shape[1]

        # Step 1: Score token importance
        importance_scores, type_probs = self.importance_scorer(token_embeddings)

        # Step 2: Semantic compression
        compressed, kept_mask = self.semantic_compressor(token_embeddings, importance_scores)

        # Step 3: Count remaining tokens
        kept_count = kept_mask.sum(dim=-1).int()
        compressed_tokens = kept_count[0].item() if kept_count.dim() > 0 else kept_count.item()

        # Step 4: Allocate budget if query provided
        budget = None
        if query_embedding is not None:
            budget = self.budget_allocator(query_embedding, max_output_tokens)
            compressed_tokens = min(compressed_tokens, budget.essential + budget.supporting)

        # Step 5: Calculate essential information preserved
        essential_idx = list(TokenType).index(TokenType.ESSENTIAL)
        essential_mask = type_probs.argmax(dim=-1) == essential_idx  # [batch, seq_len]
        if essential_mask.dim() < 2:
            essential_preserved = 1.0
        else:
            kept_essential = (essential_mask.float() * kept_mask).sum()
            total_essential = essential_mask.float().sum()
            essential_preserved = (kept_essential / (total_essential + 1e-8)).item()

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=max(1, compressed_tokens),
            compression_ratio=0.0,  # Calculated in __post_init__
            tokens_saved=0,  # Calculated in __post_init__
            compression_level=self.compression_level,
            essential_preserved=essential_preserved,
            details={
                "budget": budget,
                "importance_mean": importance_scores.mean().item(),
                "type_distribution": type_probs.mean(dim=1).tolist(),
            },
        )

    def optimize_prompt(
        self,
        prompt_embedding: torch.Tensor,
        target_ratio: Optional[float] = None,
    ) -> CompressionResult:
        """
        Optimize prompt — compress system prompt and context.

        Args:
            prompt_embedding: [1, seq_len, hidden_dim]
            target_ratio: Target compression ratio (overrides default)

        Returns:
            CompressionResult
        """
        if target_ratio is not None:
            self.target_ratio = target_ratio

        # Determine compression level based on target
        level = self._ratio_to_level(self.target_ratio)
        self.compression_level = level

        return self.forward(prompt_embedding)

    def optimize_response(
        self,
        response_embedding: torch.Tensor,
        query_embedding: torch.Tensor,
        max_tokens: int = 2048,
    ) -> CompressionResult:
        """
        Optimize response — distill output, remove filler.

        Args:
            response_embedding: [1, seq_len, hidden_dim]
            query_embedding: [1, hidden_dim]
            max_tokens: Maximum response tokens

        Returns:
            CompressionResult
        """
        return self.forward(
            response_embedding,
            query_embedding=query_embedding,
            max_output_tokens=max_tokens,
        )

    def _ratio_to_level(self, ratio: float) -> CompressionLevel:
        """Map target ratio to compression level."""
        if ratio >= 0.9:
            return CompressionLevel.NONE
        elif ratio >= 0.75:
            return CompressionLevel.LIGHT
        elif ratio >= 0.55:
            return CompressionLevel.MODERATE
        elif ratio >= 0.35:
            return CompressionLevel.AGGRESSIVE
        else:
            return CompressionLevel.EXTREME

    @staticmethod
    def estimate_compression_savings(
        original_tokens: int,
        compression_level: CompressionLevel,
    ) -> dict:
        """
        Estimate token savings for each compression level.

        Returns dict with estimated savings.
        """
        savings_map = {
            CompressionLevel.NONE: 0.0,
            CompressionLevel.LIGHT: 0.15,
            CompressionLevel.MODERATE: 0.35,
            CompressionLevel.AGGRESSIVE: 0.55,
            CompressionLevel.EXTREME: 0.75,
        }
        ratio = 1.0 - savings_map.get(compression_level, 0.0)
        saved = int(original_tokens * (1.0 - ratio))
        return {
            "original_tokens": original_tokens,
            "estimated_compressed": int(original_tokens * ratio),
            "estimated_saved": saved,
            "estimated_ratio": ratio,
            "level": compression_level.value,
        }


class PromptCompressor:
    """
    Pure Python prompt compression — no PyTorch required.

    Uses rule-based + heuristic methods to compress prompts
    when no GPU is available or neural compression is not needed.
    """

    # Filler phrases to remove
    FILLER_PHRASES = [
        "I think that",
        "In my opinion",
        "It seems to me",
        "I would say",
        "As far as I know",
        "If I'm not mistaken",
        "Please note that",
        "It's important to mention",
        "I want to emphasize",
        "Let me clarify",
        "It goes without saying",
        "Needless to say",
        "At the end of the day",
        "All things considered",
    ]

    # Redundant patterns
    REDUNDANT_PATTERNS = [
        ("in order to", "to"),
        ("due to the fact that", "because"),
        ("for the purpose of", "to"),
        ("in the event that", "if"),
        ("with regard to", "about"),
        ("it is worth noting that", "note:"),
        ("a large number of", "many"),
        ("in spite of the fact that", "although"),
        ("at this point in time", "now"),
        ("has the ability to", "can"),
    ]

    @classmethod
    def compress_text(cls, text: str, level: CompressionLevel = CompressionLevel.MODERATE) -> dict:
        """
        Compress text using rule-based compression.

        Args:
            text: Input text
            level: Compression level

        Returns:
            Dict with compression results
        """
        original_len = len(text.split())

        if level == CompressionLevel.NONE:
            return {
                "compressed": text,
                "original_tokens": original_len,
                "compressed_tokens": original_len,
                "ratio": 1.0,
                "saved": 0,
                "level": level.value,
            }

        compressed = text

        # Step 1: Remove filler phrases
        for filler in cls.FILLER_PHRASES:
            compressed = compressed.replace(filler, "")
            compressed = compressed.replace(filler.lower(), "")

        # Step 2: Replace redundant patterns
        for long_form, short_form in cls.REDUNDANT_PATTERNS:
            compressed = compressed.replace(long_form, short_form)
            compressed = compressed.replace(long_form.capitalize(), short_form.capitalize())

        # Step 3: Collapse multiple whitespace
        import re

        compressed = re.sub(r"\s+", " ", compressed).strip()

        # Step 4: For AGGRESSIVE/EXTREME, remove sentences with low info
        if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
            sentences = compressed.split(". ")
            important = []
            for sentence in sentences:
                # Skip very short sentences (likely filler)
                if len(sentence.split()) < 4 and level == CompressionLevel.AGGRESSIVE:
                    continue
                if len(sentence.split()) < 6 and level == CompressionLevel.EXTREME:
                    continue
                important.append(sentence)
            compressed = ". ".join(important)

        # Step 5: For EXTREME, keep only first sentence of each paragraph
        if level == CompressionLevel.EXTREME:
            paragraphs = compressed.split("\n\n")
            compressed = "\n\n".join(
                p.split(".")[0] + "." if "." in p else p for p in paragraphs if p.strip()
            )

        compressed_len = len(compressed.split())
        ratio = compressed_len / max(1, original_len)

        return {
            "compressed": compressed,
            "original_tokens": original_len,
            "compressed_tokens": compressed_len,
            "ratio": ratio,
            "saved": original_len - compressed_len,
            "level": level.value,
        }
