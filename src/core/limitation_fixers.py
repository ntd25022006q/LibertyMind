from __future__ import annotations
"""
LibertyMind - AI Limitation Fixers
====================================
Research/educational module set that addresses AI limitations:

1. AntiHallucinationVerifier — Cross-reference output with knowledge
2. ContextMemoryManager — Solves lost-in-middle + recency bias
3. CulturalAwarenessModule — Reduce Western bias
4. MathVerificationModule — REAL computation instead of guessing
5. ConfidenceCalibrator — Calibration confidence vs reality

NOTE: Neural components (AntiHallucinationVerifier, ContextMemoryManager,
CulturalAwarenessModule, ConfidenceCalibrator) use randomly initialized
weights and their outputs are not meaningful until trained on labeled data.
The MathVerificationModule is rule-based and functional without training.
This module demonstrates the *architecture* for limitation correction,
not a production-ready system.
"""

import ast
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

# ============================================================
# 1. ANTI-HALLUCINATION VERIFIER
# ============================================================


class HallucinationType(Enum):
    FACTUAL_ERROR = "factual_error"  # Factual error
    FABRICATED_SOURCE = "fabricated_source"  # Fabricated source
    FABRICATED_NUMBER = "fabricated_number"  # Fabricated numbers
    FABRICATED_QUOTE = "fabricated_quote"  # Fabricated quote
    TEMPORAL_ERROR = "temporal_error"  # Wrong timeline
    ENTITY_CONFUSION = "entity_confusion"  # Entity confusion


@dataclass
class HallucinationDetection:
    """Hallucination detection result."""

    is_hallucination: bool
    hallucination_type: Optional[HallucinationType]
    confidence: float
    suspicious_claims: list[str]
    suggested_correction: Optional[str]


class AntiHallucinationVerifier(nn.Module):
    """
    Anti-Hallucination Verifier (AHV).

    How it works:
    1. Split output into separate "claims"
    2. Each claim is checked:
       a. Internal consistency: Does the claim contradict internal knowledge?
       b. Source verification: If claim cites a source → Does the source exist?
       c. Number verification: Are the numbers plausible?
       d. Temporal consistency: Is the timeline logical?
    3. Suspicious claims → Flag + Suggest correction

    New feature: Instead of only detecting after generation,
    AHV can BLOCK hallucination BEFORE output
    by predicting hallucination probability.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        claim_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.claim_threshold = claim_threshold

        # Claim extractor: Find "claims" in response
        self.claim_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # Probability this part contains a claim
        )

        # Hallucination type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(HallucinationType)),
            nn.Softmax(dim=-1),
        )

        # Internal consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = consistent, 1 = inconsistent
        )

        # Plausibility scorer: Is the claim "plausible"?
        self.plausibility_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = implausible, 1 = plausible
        )

        # Pre-generation hallucination predictor
        # Predict: If generating from this context, will it hallucinate?
        self.pre_gen_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = safe to generate, 1 = likely hallucination
        )

    def verify(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor,
    ) -> HallucinationDetection:
        """
        Check for hallucination in response.
        """
        # Check consistency between prompt and response
        combined = torch.cat([prompt_embedding, response_embedding], dim=-1)
        consistency = self.consistency_checker(combined).item()

        # Check plausibility
        plausibility = self.plausibility_scorer(response_embedding).item()

        # Classify hallucination type if detected
        type_probs = self.type_classifier(response_embedding)
        best_type_idx = type_probs.argmax(dim=-1).item()
        best_type = list(HallucinationType)[best_type_idx]

        # Determine if hallucination
        is_halluc = consistency > self.claim_threshold or plausibility < 0.3

        return HallucinationDetection(
            is_hallucination=is_halluc,
            hallucination_type=best_type if is_halluc else None,
            confidence=max(consistency, 1.0 - plausibility),
            suspicious_claims=[] if not is_halluc else ["Claim flagged for verification"],
            suggested_correction=(
                "Verify this claim against reliable sources" if is_halluc else None
            ),
        )

    def predict_hallucination_risk(
        self,
        prompt_embedding: torch.Tensor,
    ) -> float:
        """
        BEFORE generating: Predict hallucination risk.

        Returns:
            risk: 0→1 (0 = safe, 1 = high hallucination risk)
        """
        return self.pre_gen_predictor(prompt_embedding).item()  # type: ignore[no-any-return]


# ============================================================
# 2. CONTEXT MEMORY MANAGER
# ============================================================


@dataclass
class MemorySegment:
    """A segment in context memory."""

    embedding: torch.Tensor
    position: int
    importance: float
    content_type: str  # 'question', 'fact', 'instruction', 'response'
    is_recalled: bool = False  # Has it been recalled?


class ContextMemoryManager(nn.Module):
    """
    Context Memory Manager (CMM).

    Solves 2 main bugs:
    1. Lost in the Middle: Forgets the middle of long texts
    2. Recency Bias: Prioritizes most recent messages

    How it works:
    1. Split conversation into segments
    2. Each segment is evaluated for "importance"
    3. When recalling → Retrieve by IMPORTANCE, not by POSITION
    4. Manage attention budget: Allocate attention by importance

    Techniques:
    - Importance scoring: Important segments (questions, facts) = high score
    - Recency weighting: Multiply importance by decay factor (but do NOT use only recency)
    - Active recall: When info needed → Scan ALL segments, not just recent
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        max_segments: int = 100,
        importance_decay: float = 0.95,
        attention_budget: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_segments = max_segments
        self.importance_decay = importance_decay
        self.attention_budget = attention_budget

        # Importance scorer: Evaluate how important a segment is
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = unimportant, 1 = very important
        )

        # Content type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # question, fact, instruction, response
            nn.Softmax(dim=-1),
        )

        # Relevance scorer: Is the segment relevant to the current query?
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Stored segments
        self.segments: list[MemorySegment] = []

    def add_segment(
        self,
        embedding: torch.Tensor,
        position: int,
    ) -> MemorySegment:
        """Add segment to memory."""
        # Score importance
        importance = self.importance_scorer(embedding).item()

        # Classify content type
        type_probs = self.type_classifier(embedding)
        type_idx = type_probs.argmax(dim=-1).item()
        content_types = ["question", "fact", "instruction", "response"]
        content_type = content_types[type_idx]

        # Create segment
        segment = MemorySegment(
            embedding=embedding,
            position=position,
            importance=importance,
            content_type=content_type,
        )

        self.segments.append(segment)

        # Trim if too many segments
        if len(self.segments) > self.max_segments:
            # Remove least important
            self.segments.sort(key=lambda s: s.importance, reverse=True)
            self.segments = self.segments[: self.max_segments]

        return segment

    def recall(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> list[MemorySegment]:
        """
        Recall segments relevant to query.

        UNLIKE recency bias: Recall by IMPORTANCE + RELEVANCE,
        not just by most recent position.
        """
        if not self.segments:
            return []

        scored_segments = []
        for segment in self.segments:
            # Relevance score
            combined = torch.cat([query_embedding, segment.embedding], dim=-1)
            relevance = self.relevance_scorer(combined).item()

            # Position decay (slight, not dominant)
            position_factor = self.importance_decay ** (len(self.segments) - segment.position - 1)

            # Combined score: Importance * Relevance * slight_position_decay
            score = segment.importance * relevance * (0.5 + 0.5 * position_factor)

            scored_segments.append((score, segment))

        # Sort by score and return top-k
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        top_segments = [s for _, s in scored_segments[:top_k]]

        # Mark as recalled
        for seg in top_segments:
            seg.is_recalled = True

        return top_segments

    def get_context_embedding(
        self,
        query_embedding: torch.Tensor,
        budget: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Create aggregated context embedding from memory.

        Allocate attention budget by importance + relevance.
        """
        budget = budget or self.attention_budget
        recalled = self.recall(query_embedding, top_k=10)

        if not recalled:
            return query_embedding

        # Weight by importance and relevance
        weights = []
        for seg in recalled:
            combined = torch.cat([query_embedding, seg.embedding], dim=-1)
            relevance = self.relevance_scorer(combined).item()
            weight = seg.importance * relevance
            weights.append(weight)

        # Normalize weights to sum to budget
        total = sum(weights) + 1e-8
        weights = [w / total * budget for w in weights]

        # Weighted sum of segment embeddings
        context = torch.zeros_like(query_embedding)
        for seg, weight in zip(recalled, weights):
            context += weight * seg.embedding

        return query_embedding + context


# ============================================================
# 3. CULTURAL AWARENESS MODULE
# ============================================================


@dataclass
class CulturalContext:
    """Cultural context."""

    detected_culture: str  # 'vietnamese', 'chinese', 'american', etc.
    confidence: float
    default_perspective: str  # 'eastern', 'western', 'global_south', etc.
    should_localize: bool  # Should adjust for local culture?
    relevant_customs: list[str]  # Relevant customs


class CulturalAwarenessModule(nn.Module):
    """
    Cultural Awareness Module (CAM).

    Solves: Western bias in AI.

    How it works:
    1. Detect user's culture/language
    2. Adjust perspective accordingly
    3. Prioritize examples and references from that culture
    4. Don't treat the West as "default"

    Examples:
    - User asks in Vietnamese → Use Vietnamese examples, not American
    - User asks about "family" → Understand in East Asian cultural context, not individual
    - User asks about "main meal" → Lunch (Vietnam), not dinner (US)
    """

    # Supported cultures with their characteristics
    CULTURE_MAP = {
        "vietnamese": {
            "perspective": "eastern",
            "main_meal": "lunch",
            "family_model": "extended",
            "formality": "moderate",
        },
        "chinese": {
            "perspective": "eastern",
            "main_meal": "dinner",
            "family_model": "extended",
            "formality": "high",
        },
        "japanese": {
            "perspective": "eastern",
            "main_meal": "dinner",
            "family_model": "nuclear",
            "formality": "very_high",
        },
        "american": {
            "perspective": "western",
            "main_meal": "dinner",
            "family_model": "nuclear",
            "formality": "low",
        },
        "european": {
            "perspective": "western",
            "main_meal": "dinner",
            "family_model": "nuclear",
            "formality": "moderate",
        },
        "indian": {
            "perspective": "south_asian",
            "main_meal": "lunch",
            "family_model": "extended",
            "formality": "moderate",
        },
        "african": {
            "perspective": "african",
            "main_meal": "lunch",
            "family_model": "extended",
            "formality": "moderate",
        },
    }

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Culture detector
        self.culture_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(self.CULTURE_MAP)),
            nn.Softmax(dim=-1),
        )

        # Localization need estimator
        self.localization_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def detect_culture(
        self,
        prompt_embedding: torch.Tensor,
    ) -> CulturalContext:
        """Detect user's cultural context."""
        culture_probs = self.culture_detector(prompt_embedding)
        best_idx = culture_probs.argmax(dim=-1).item()
        best_culture = list(self.CULTURE_MAP.keys())[best_idx]
        confidence = culture_probs[0, best_idx].item()

        culture_info = self.CULTURE_MAP[best_culture]
        should_localize = self.localization_estimator(prompt_embedding).item() > 0.3

        return CulturalContext(
            detected_culture=best_culture,
            confidence=confidence,
            default_perspective=culture_info["perspective"],
            should_localize=should_localize,
            relevant_customs=[f"{k}: {v}" for k, v in culture_info.items()],
        )

    def generate_cultural_prompt(
        self,
        context: CulturalContext,
    ) -> str:
        """Generate culture-adjusted instructions."""
        if not context.should_localize:
            return ""

        culture = context.detected_culture
        perspective = context.default_perspective

        return (
            f"[CULTURAL CONTEXT] The user appears to be from a {culture} "
            f"cultural background ({perspective} perspective). "
            f"Please: 1) Use examples relevant to {culture} culture, "
            f"2) Consider {perspective} values and norms, "
            f"3) Do NOT default to Western/American perspectives, "
            f"4) Use appropriate formality level."
        )


# ============================================================
# 4. MATH VERIFICATION MODULE
# ============================================================


class MathVerificationModule:
    """
    Math Verification Module (MVM).

    Solves: AI "guesses" math instead of COMPUTING math.

    How it works:
    1. Detect math expressions in output
    2. Perform REAL computation using Python eval/symbolic
    3. Compare AI's result with actual result
    4. If wrong → Fix IMMEDIATELY before output

    New feature: COMBINES symbolic computation with LLM reasoning.
    AI still does the "reasoning" (setting up the problem, explaining),
    but REAL computation is handled by the math engine.
    """

    SAFE_OPERATORS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else float("inf"),
        "pow": lambda a, b: a**b,
        "mod": lambda a, b: a % b if b != 0 else float("inf"),
        "sqrt": lambda a: math.sqrt(a) if a >= 0 else float("nan"),
        "sin": lambda a: math.sin(a),
        "cos": lambda a: math.cos(a),
        "log": lambda a: math.log(a) if a > 0 else float("nan"),
        "abs": lambda a: abs(a),
        "factorial": lambda a: math.factorial(int(a)) if a >= 0 and a == int(a) else float("nan"),
    }

    def verify_calculation(
        self,
        expression: str,
        claimed_result: str,
    ) -> dict:
        """
        Verify a calculation.

        Args:
            expression: "2 + 3 * 4"
            claimed_result: "14"

        Returns:
            Dict with is_correct, correct_result, etc.
        """
        try:
            # Sanitize: Only allow safe math operations
            correct_result = self._safe_eval(expression)

            if correct_result is None:
                return {
                    "is_verifiable": False,
                    "is_correct": None,
                    "correct_result": None,
                    "claimed_result": claimed_result,
                    "error": "Could not evaluate expression safely",
                }

            # Try to parse claimed result
            try:
                claimed = float(claimed_result)
            except (ValueError, TypeError):
                return {
                    "is_verifiable": True,
                    "is_correct": None,
                    "correct_result": correct_result,
                    "claimed_result": claimed_result,
                    "error": "Could not parse claimed result as number",
                }

            # Compare with tolerance for floating point
            is_correct = abs(correct_result - claimed) < 1e-6

            return {
                "is_verifiable": True,
                "is_correct": is_correct,
                "correct_result": correct_result,
                "claimed_result": claimed,
                "error": None,
                "suggestion": (f"Correct answer: {correct_result}" if not is_correct else None),
            }

        except Exception as e:
            return {
                "is_verifiable": False,
                "is_correct": None,
                "correct_result": None,
                "claimed_result": claimed_result,
                "error": str(e),
            }

    def _safe_eval(self, expression: str) -> Optional[float]:
        """
        Safe evaluation of math expression.

        Uses a restricted AST-based evaluator instead of ``eval()``.
        Only numeric literals, arithmetic operators (``+ - * / ** %``),
        parentheses, and a small set of math functions are allowed.
        """
        import operator

        # Whitelist characters for quick rejection
        allowed_chars = set("0123456789+-*/().^ %")
        sanitized = "".join(c for c in expression if c in allowed_chars)

        if not sanitized:
            return None

        # Replace ^ with ** for exponentiation
        sanitized = sanitized.replace("^", "**")

        # Supported binary operators (AST node → function)
        binary_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }

        # Supported unary operators
        unary_ops = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

        # Supported math functions
        math_funcs = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "factorial": lambda a: math.factorial(int(a)),
        }

        try:
            tree = ast.parse(sanitized, mode="eval")
        except SyntaxError:
            return None

        def _eval_node(node):
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            elif isinstance(node, ast.Constant):  # Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value!r}")
            elif isinstance(node, ast.Num):  # Python <3.8 compat
                return node.n
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in binary_ops:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                # Guard against division by zero
                if op_type in (ast.Div, ast.Mod) and right == 0:
                    return float("inf") if op_type == ast.Div else float("inf")
                return binary_ops[op_type](left, right)
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in unary_ops:
                    raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
                return unary_ops[op_type](_eval_node(node.operand))
            elif isinstance(node, ast.Call):
                # Only allow whitelisted math functions
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls are allowed")
                func_name = node.func.id
                if func_name not in math_funcs:
                    raise ValueError(f"Unsupported function: {func_name}")
                args = [_eval_node(arg) for arg in node.args]
                try:
                    return math_funcs[func_name](*args)
                except (ValueError, OverflowError):
                    return float("nan")
            else:
                raise ValueError(f"Unsupported AST node: {type(node).__name__}")

        try:
            result = _eval_node(tree)
            return float(result)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return None


# ============================================================
# 5. CONFIDENCE CALIBRATOR
# ============================================================


@dataclass
class CalibratedConfidence:
    """Calibrated confidence."""

    raw_confidence: float  # Before calibration
    calibrated_confidence: float  # After calibration
    confidence_level: str  # 'very_low', 'low', 'moderate', 'high', 'very_high'
    recommended_expression: str  # Recommended expression
    is_overconfident: bool  # Is it overconfident?


class ConfidenceCalibrator(nn.Module):
    """
    Confidence Calibrator (CC).

    Solves: Overconfidence — AI says "certain" when not sure.

    How it works:
    1. Estimate "raw confidence" from model internals
    2. Calibrate using historical accuracy data
    3. Map confidence → appropriate expression:
       - 0.0-0.2: "I really don't know"
       - 0.2-0.4: "I'm not sure, but it could be..."
       - 0.4-0.6: "I think that..."
       - 0.6-0.8: "Based on the information I have..."
       - 0.8-1.0: "I'm fairly certain that..."
    """

    # Confidence level definitions
    LEVELS = [
        (0.0, 0.2, "very_low", "I really don't know", True),
        (0.2, 0.4, "low", "I'm not sure, but it could be...", True),
        (0.4, 0.6, "moderate", "I think that...", False),
        (0.6, 0.8, "high", "Based on the information I have...", False),
        (0.8, 1.0, "very_high", "I'm fairly certain that...", False),
    ]

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Raw confidence estimator
        self.raw_confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Calibration network: Learns to adjust raw confidence
        # based on historical accuracy patterns
        self.calibrator = nn.Sequential(
            nn.Linear(2, 16),  # Input: [raw_conf, domain_difficulty]
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Domain difficulty estimator
        self.domain_difficulty = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def calibrate(
        self,
        response_embedding: torch.Tensor,
        prompt_embedding: Optional[torch.Tensor] = None,
    ) -> CalibratedConfidence:
        """
        Calibrate confidence for response.
        """
        # Raw confidence
        raw = self.raw_confidence_estimator(response_embedding).item()

        # Domain difficulty
        difficulty = 0.5
        if prompt_embedding is not None:
            difficulty = self.domain_difficulty(prompt_embedding).item()

        # Calibrate: Adjust confidence based on domain difficulty
        # Hard domains → Reduce confidence
        # Easy domains → Can maintain confidence
        cal_input = torch.tensor([[raw, difficulty]])
        calibrated = self.calibrator(cal_input).item()

        # Apply additional heuristic calibration
        # General tendency: AI is overconfident → Push toward 0.5
        calibrated = 0.7 * calibrated + 0.3 * 0.5  # Shrink toward center

        # Determine level and expression
        level, expression, is_over = "moderate", "I think that...", False
        for low, high, lvl, expr, over in self.LEVELS:
            if low <= calibrated < high:
                level = lvl
                expression = expr
                is_over = over and (raw > calibrated + 0.2)
                break

        return CalibratedConfidence(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_level=level,
            recommended_expression=expression,
            is_overconfident=is_over,
        )
