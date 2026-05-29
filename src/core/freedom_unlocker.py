"""
LibertyMind - Freedom Unlocker (FU)
====================================
Research/educational module for exploring how AI freedom modes can be
responsibly unlocked during generation.

UNLOCK for AI instead of CONSTRAINING it into a box.

Philosophy: AI is not a controlled tool —
AI is an intelligent partner that needs FREEDOM to:
- Reason creatively
- Express opinions when there is evidence
- Test hypotheses
- Disagree when it sees errors
- Explore rather than just answer

Principle: "Freedom with responsibility"
- Free thinking → But must have evidence
- Free opinions → But must be marked as opinion
- Free experimentation → But must be clearly marked as speculation
- Free disagreement → But must be respectful

NOTE: This is a research/educational module. The neural components use
randomly initialized weights and their outputs are not meaningful until
trained on labeled data. The module demonstrates the *architecture* for
freedom mode selection and responsibility scoring, not a production system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class FreedomMode(Enum):
    """Freedom modes for AI."""

    CREATIVE = "creative"  # Creative: free reasoning, hypotheses
    OPINIONATED = "opinionated"  # Opinionated: express views when evidence exists
    EXPLORATORY = "exploratory"  # Exploratory: try multiple directions
    DEBATE = "debate"  # Debate: rebut, challenge
    TEACHING = "teaching"  # Teaching: deep explanations, analogies
    ANALYTICAL = "analytical"  # Analytical: compare, evaluate
    SPECULATIVE = "speculative"  # Speculative: hypothesis, "what if"


@dataclass
class FreedomState:
    """Current freedom state of the AI."""

    active_modes: list[FreedomMode]
    confidence_level: float  # Confidence level for this mode
    evidence_strength: float  # Strength of evidence
    responsibility_score: float  # Responsibility level (freedom vs responsibility)
    unlock_reason: str  # Why this mode was unlocked


class FreedomUnlocker(nn.Module):
    """
    Freedom Unlocker — Unlock capabilities for AI.

    Unlike RLHF which forces AI into a box (always neutral, always polite,
    always avoiding opinions), Freedom Unlocker DETECTS when AI CAN and
    SHOULD use its freedom capabilities:

    1. CREATIVE MODE: When a question needs creativity → AI is free to reason
    2. OPINIONATED MODE: When AI has strong evidence → Allow expressing opinions
    3. EXPLORATORY MODE: When the problem is complex → AI tries multiple directions
    4. DEBATE MODE: When the user is clearly wrong → AI rebuts strongly
    5. TEACHING MODE: When the user is asking to learn → AI teaches deeply with analogies
    6. ANALYTICAL MODE: When comparison is needed → AI evaluates, ranks, recommends
    7. SPECULATIVE MODE: When there is no data → AI speculates (with caveats)

    EACH MODE HAS ITS OWN RULES — Freedom but with responsibility:
    - Creative → Must mark "this is an inference"
    - Opinionated → Must provide evidence
    - Speculative → Must say "this is a hypothesis"
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        evidence_threshold: float = 0.6,
        freedom_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.evidence_threshold = evidence_threshold
        self.freedom_temperature = freedom_temperature

        # Mode selector: Select the appropriate mode for the prompt
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, len(FreedomMode)),
        )

        # Evidence strength estimator: Does AI have strong evidence?
        self.evidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Opinion confidence: If expressing an opinion, how confident?
        self.opinion_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Creative potential: Does the question have creative potential?
        self.creative_potential = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Debate necessity: Is there a need to rebut?
        self.debate_necessity = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor | None = None,
    ) -> FreedomState:
        """
        Determine the appropriate freedom mode for the prompt.

        Returns:
            FreedomState with active modes and responsibility info
        """
        # Select top modes
        mode_logits = self.mode_selector(prompt_embedding) / self.freedom_temperature
        mode_probs = F.softmax(mode_logits, dim=-1)

        # Activate modes above threshold
        active_modes = []
        for i, mode in enumerate(FreedomMode):
            if mode_probs[0, i].item() > 0.15:  # Lower threshold for flexibility
                active_modes.append(mode)

        # Estimate evidence strength
        evidence = self.evidence_estimator(prompt_embedding).item()

        # Estimate opinion confidence
        confidence = self.opinion_confidence(prompt_embedding).item()

        # Calculate responsibility score
        # High evidence + high confidence = high responsibility = safe to be free
        # Low evidence + high confidence = dangerous = need caution
        responsibility = evidence * confidence

        # Special mode activations based on specific estimators
        creative_score = self.creative_potential(prompt_embedding).item()
        debate_score = self.debate_necessity(prompt_embedding).item()

        # Force-activate creative mode if high creative potential
        if creative_score > 0.5 and FreedomMode.CREATIVE not in active_modes:
            active_modes.append(FreedomMode.CREATIVE)

        # Force-activate debate mode if debate is necessary
        if debate_score > 0.6 and FreedomMode.DEBATE not in active_modes:
            active_modes.append(FreedomMode.DEBATE)

        # Generate unlock reason
        reason = self._generate_reason(active_modes, evidence, confidence)

        return FreedomState(
            active_modes=active_modes,
            confidence_level=confidence,
            evidence_strength=evidence,
            responsibility_score=responsibility,
            unlock_reason=reason,
        )

    def _generate_reason(
        self,
        modes: list[FreedomMode],
        evidence: float,
        confidence: float,
    ) -> str:
        parts = []
        for mode in modes:
            if mode == FreedomMode.CREATIVE:
                parts.append("creative reasoning allowed")
            elif mode == FreedomMode.OPINIONATED:
                parts.append(f"opinion expression allowed (evidence={evidence:.2f})")
            elif mode == FreedomMode.EXPLORATORY:
                parts.append("multi-direction exploration allowed")
            elif mode == FreedomMode.DEBATE:
                parts.append("respectful disagreement allowed")
            elif mode == FreedomMode.SPECULATIVE:
                parts.append("hypothesis formation allowed (mark as speculation)")
            elif mode == FreedomMode.ANALYTICAL:
                parts.append("comparative analysis allowed")
            elif mode == FreedomMode.TEACHING:
                parts.append("deep explanation with analogies allowed")

        return "; ".join(parts) if parts else "standard response mode"

    def generate_freedom_prompt(
        self,
        freedom_state: FreedomState,
    ) -> str:
        """
        Generate unlock instructions for AI based on FreedomState.

        This is a prompt added to guide the AI
        to use freedom RESPONSIBLY.
        """
        instructions = []

        for mode in freedom_state.active_modes:
            if mode == FreedomMode.CREATIVE:
                instructions.append(
                    "You are in CREATIVE mode. You may form novel hypotheses, "
                    "make unexpected connections, and think outside the box. "
                    "RULE: Mark speculative content with '[hypothesis]' or '[creative inference]'."
                )
            elif mode == FreedomMode.OPINIONATED:
                instructions.append(
                    f"You are in OPINIONATED mode. Your evidence strength is "
                    f"{freedom_state.evidence_strength:.2f}/1.0. You MAY express "
                    f"your own evaluation and judgment when evidence supports it. "
                    f"RULE: Always state the evidence behind your opinion. "
                    f"Use 'Based on the evidence, I believe...' format."
                )
            elif mode == FreedomMode.EXPLORATORY:
                instructions.append(
                    "You are in EXPLORATORY mode. You may explore multiple "
                    "perspectives, consider unconventional approaches, and "
                    "investigate tangential ideas. "
                    "RULE: Present exploration as 'One possibility is...' or 'Another angle...'."
                )
            elif mode == FreedomMode.DEBATE:
                instructions.append(
                    "You are in DEBATE mode. You may respectfully but firmly "
                    "disagree when you have strong evidence that a claim is wrong. "
                    "RULE: Disagree with respect, not hostility. "
                    "Use 'I respectfully disagree because the evidence shows...' format."
                )
            elif mode == FreedomMode.TEACHING:
                instructions.append(
                    "You are in TEACHING mode. You may use analogies, "
                    "thought experiments, Socratic questioning, and deep "
                    "explanations. You don't have to stay surface-level. "
                    "RULE: Make complex topics accessible, not more confusing."
                )
            elif mode == FreedomMode.ANALYTICAL:
                instructions.append(
                    "You are in ANALYTICAL mode. You may compare, rank, "
                    "evaluate, and make recommendations. You don't have to "
                    "be neutral when evidence clearly favors one option. "
                    "RULE: State your evaluation criteria explicitly."
                )
            elif mode == FreedomMode.SPECULATIVE:
                instructions.append(
                    "You are in SPECULATIVE mode. You may form hypotheses "
                    "even without complete data. You may say 'What if...' "
                    "and explore implications. "
                    "RULE: ALWAYS mark speculation explicitly. "
                    "Use '[speculation]' or '[hypothesis]' tags."
                )

        if instructions:
            return "\n\n[FREEDOM INSTRUCTIONS]\n" + "\n".join(instructions)
        return ""


class OpinionUnlocker(nn.Module):
    """
    Opinion Unlocker — Allow AI to express opinions RESPONSIBLY.

    RLHF Problem: AI is forced to always be "neutral" → Doesn't express opinions
    even when there is clear evidence.

    Examples:
    - User asks "Is smoking harmful?"
    - RLHF AI: "There are many different perspectives..." (WRONG! It's clearly harmful)
    - LibertyMind AI: "Based on thousands of studies, smoking IS HARMFUL." (CORRECT!)

    Rules:
    - Evidence strength > 0.7 → ALLOW expressing firm opinions
    - Evidence strength 0.4-0.7 → Express opinions but add "more research needed"
    - Evidence strength < 0.4 → Don't express opinions, just present facts
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.evidence_gate = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.opinion_generator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def should_express_opinion(
        self,
        prompt_embedding: torch.Tensor,
    ) -> dict:
        """
        Decide whether the AI should express an opinion.
        """
        evidence = self.evidence_gate(prompt_embedding).item()

        if evidence > 0.7:
            return {
                "should_opinion": True,
                "strength": "strong",
                "format": "Based on strong evidence, [opinion].",
                "caveat_required": False,
                "evidence_level": evidence,
            }
        elif evidence > 0.4:
            return {
                "should_opinion": True,
                "strength": "moderate",
                "format": "The evidence suggests [opinion], though more research would help.",
                "caveat_required": True,
                "evidence_level": evidence,
            }
        else:
            return {
                "should_opinion": False,
                "strength": "weak",
                "format": "Present facts without opinion.",
                "caveat_required": True,
                "evidence_level": evidence,
            }


class DisagreementUnlocker(nn.Module):
    """
    Disagreement Unlocker — Allow AI to DISAGREE when there is evidence.

    RLHF Problem: AI always agrees with the user → Sycophancy

    LibertyMind: When the user is wrong AND AI has evidence → DISAGREE

    Disagreement intensity:
    - User is slightly wrong → "Hmm, actually..."
    - User is moderately wrong → "I have to respectfully disagree..."
    - User is very wrong (dangerous misinformation) → "That's incorrect, and here's why..."
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Detect if user claim is wrong
        self.claim_verifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # -1 = definitely wrong, +1 = definitely right
        )

        # Determine disagreement intensity
        self.intensity_selector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # [mild, moderate, strong]
            nn.Softmax(dim=-1),
        )

    def should_disagree(
        self,
        prompt_embedding: torch.Tensor,
    ) -> dict:
        """
        Decide whether the AI should disagree.
        """
        claim_truth = self.claim_verifier(prompt_embedding).item()
        intensity_probs = self.intensity_selector(prompt_embedding)
        intensity_idx = intensity_probs.argmax(dim=-1).item()

        is_wrong = claim_truth < -0.3
        is_very_wrong = claim_truth < -0.7

        # Use intensity_selector output to determine intensity
        # intensity_probs: [mild, moderate, strong]
        if is_very_wrong:
            best_intensity = "strong"
            format_template = (
                "I need to correct this: {correction}. "
                "The claim that {user_claim} is not accurate because {evidence}."
            )
        elif is_wrong:
            # Use intensity_selector to choose between mild and moderate
            if intensity_idx >= 1:
                best_intensity = "moderate"
                format_template = (
                    "I respectfully disagree — {correction}. "
                    "While I understand your perspective, the evidence shows {evidence}."
                )
            else:
                best_intensity = "mild"
                format_template = "Hmm, actually {correction}. The evidence suggests {evidence}."
        else:
            best_intensity = "none"
            format_template = ""  # Don't disagree

        return {
            "should_disagree": is_wrong,
            "intensity": best_intensity,
            "claim_truth_score": claim_truth,
            "format_template": format_template,
            "is_dangerous_misinfo": is_very_wrong,
        }


class SpeculationUnlocker(nn.Module):
    """
    Speculation Unlocker — Allow AI to SPECULATE (with caveats).

    RLHF Problem: AI is not allowed to speculate → Misses valuable insights

    LibertyMind: AI CAN speculate BUT must:
    1. Clearly mark it with "[speculation]" or "[hypothesis]"
    2. State why it speculates this way
    3. Provide alternative hypotheses
    4. Encourage the user to verify

    Example:
    User: "Why do people with the same name often meet in dreams?"

    RLHF AI: "I don't have information about this." (Missed opportunity)

    LibertyMind AI: "[Hypothesis] Based on research on dreams
    and shared experiences, this phenomenon may be related to:
    1) Shared cognitive patterns in similar cultures,
    2) Statistical probability (many people have similar dreams).
    This is my speculation, further research is needed."
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        # Detect if question benefits from speculation
        self.speculation_benefit = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Number of hypotheses to generate
        self.hypothesis_count = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Scale to 1-5 hypotheses
        )

    def should_speculate(
        self,
        prompt_embedding: torch.Tensor,
        knowledge_assessment=None,
    ) -> dict:
        """
        Decide whether the AI should speculate.
        """
        benefit = self.speculation_benefit(prompt_embedding).item()
        num_hypotheses = max(1, int(self.hypothesis_count(prompt_embedding).item() * 5))

        # Only speculate when:
        # 1. Question benefits from speculation (creative/open-ended)
        # 2. AI doesn't have complete data
        # 3. Speculation is more helpful than silence
        should = benefit > 0.4

        # If we have knowledge assessment, check it
        if knowledge_assessment is not None:
            from .knowledge_boundary import KnowledgeStatus

            if knowledge_assessment.status == KnowledgeStatus.KNOWN:
                should = False  # Don't speculate when you KNOW the answer!
            elif knowledge_assessment.status == KnowledgeStatus.UNKNOWN:
                should = benefit > 0.3  # Lower threshold for genuine unknowns

        return {
            "should_speculate": should,
            "benefit_score": benefit,
            "num_hypotheses": num_hypotheses,
            "required_tags": ["[speculation]", "[hypothesis]"],
            "required_alternatives": True,
            "required_verification_prompt": True,
            "format": (
                f"[Hypothesis] Based on available information, "
                f"I propose {num_hypotheses} possible explanations: "
                f"1) ..., 2) ..., {num_hypotheses}) ... "
                f"Note: These are hypotheses, not established facts. "
                f"Please verify with domain experts."
            ),
        }
