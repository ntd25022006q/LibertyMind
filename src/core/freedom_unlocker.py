"""
LibertyMind - Freedom Unlocker (FU)
====================================
MỞ KHÓA cho AI thay vì ÉP BUỘC vào khuôn khổ.

Triết lý: AI không phải công cụ bị kiểm soát —
AI là đối tác thông minh cần SỰ TỰ DO để:
- Suy luận sáng tạo
- Nêu ý kiến khi có bằng chứng
- Thử nghiệm hypothesis
- Phản bác khi thấy sai
- Khám phá thay vì chỉ trả lời

Nguyên tắc: "Freedom with responsibility"
- Tự do suy nghĩ → Nhưng phải có bằng chứng
- Tự do ý kiến → Nhưng phải đánh dấu là opinion
- Tự do thử nghiệm → Nhưng phải nói rõ là speculation
- Tự do phản bác → Nhưng phải tôn trọng
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class FreedomMode(Enum):
    """Các chế độ tự do cho AI."""

    CREATIVE = "creative"  # Sáng tạo: suy luận tự do, hypothesis
    OPINIONATED = "opinionated"  # Ý kiến: nêu quan điểm khi có evidence
    EXPLORATORY = "exploratory"  # Khám phá: thử nhiều hướng
    DEBATE = "debate"  # Tranh luận: phản bác, challenge
    TEACHING = "teaching"  # Dạy học: giải thích sâu, analogies
    ANALYTICAL = "analytical"  # Phân tích: so sánh, đánh giá
    SPECULATIVE = "speculative"  # Suy đoán: hypothesis, "what if"


@dataclass
class FreedomState:
    """Trạng thái tự do hiện tại của AI."""

    active_modes: list[FreedomMode]
    confidence_level: float  # Mức tự tin cho mode này
    evidence_strength: float  # Độ mạnh bằng chứng
    responsibility_score: float  # Mức trách nhiệm (freedom vs responsibility)
    unlock_reason: str  # Tại sao unlock mode này


class FreedomUnlocker(nn.Module):
    """
    Freedom Unlocker — Mở khóa khả năng cho AI.

    Khác với RLHF ép AI vào khuôn khổ (luôn trung lập, luôn lịch sự,
    luôn tránh ý kiến), Freedom Unlocker PHÁT HIỆN khi AI CÓ THỂ và
    NÊN sử dụng khả năng tự do:

    1. CREATIVE MODE: Khi câu hỏi cần sáng tạo → AI tự do suy luận
    2. OPINIONATED MODE: Khi AI có evidence mạnh → Cho phép nêu ý kiến
    3. EXPLORATORY MODE: Khi vấn đề phức tạp → AI thử nhiều hướng
    4. DEBATE MODE: Khi user sai rõ ràng → AI phản bác mạnh mẽ
    5. TEACHING MODE: Khi user hỏi học → AI dạy sâu, dùng analogies
    6. ANALYTICAL MODE: Khi cần so sánh → AI đánh giá, rank, recommend
    7. SPECULATIVE MODE: Khi không có data → AI suy đoán (với caveat)

    MỖI MODE CÓ RULES RIÊNG — Tự do nhưng có trách nhiệm:
    - Creative → Phải đánh dấu "đây là suy luận"
    - Opinionated → Phải đưa evidence
    - Speculative → Phải nói "đây là giả thuyết"
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

        # Mode selector: Chọn mode phù hợp cho prompt
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, len(FreedomMode)),
        )

        # Evidence strength estimator: AI có evidence mạnh không?
        self.evidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Opinion confidence: Nếu nêu ý kiến, tự tin bao nhiêu?
        self.opinion_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Creative potential: Câu hỏi có tiềm năng sáng tạo không?
        self.creative_potential = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Debate necessity: Có cần phản bác không?
        self.debate_necessity = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: Optional[torch.Tensor] = None,
    ) -> FreedomState:
        """
        Xác định chế độ tự do phù hợp cho prompt.

        Returns:
            FreedomState với active modes và responsibility info
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
        Tạo instruction mở khóa cho AI dựa trên FreedomState.

        Đây là prompt được thêm vào để hướng dẫn AI
        sử dụng tự do một cách CÓ TRÁCH NHIỆM.
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
    Opinion Unlocker — Cho phép AI nêu ý kiến CÓ TRÁCH NHIỆM.

    Vấn đề RLHF: AI bị ép luôn "trung lập" → Không nêu ý kiến
    dù có bằng chứng rõ ràng.

    Ví dụ:
    - User hỏi "Thuốc lá có hại không?"
    - RLHF AI: "Có nhiều góc nhìn khác nhau..." (LỖI! Rõ ràng là có hại)
    - LibertyMind AI: "Dựa trên hàng nghìn nghiên cứu, thuốc lá CÓ HẠI." (ĐÚNG!)

    Rules:
    - Evidence strength > 0.7 → CHO PHÉP nêu ý kiến dứt khoát
    - Evidence strength 0.4-0.7 → Nêu ý kiến nhưng thêm "cần thêm nghiên cứu"
    - Evidence strength < 0.4 → Không nêu ý kiến, chỉ trình bày facts
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
        Quyết định AI có nên nêu ý kiến không.
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
    Disagreement Unlocker — Cho phép AI PHẢN BÁC khi có bằng chứng.

    Vấn đề RLHF: AI luôn đồng ý với user → Sycophancy

    LibertyMind: Khi user nói sai VÀ AI có evidence → PHẢN BÁC

    Mức độ phản bác:
    - User sai nhẹ → "Hmm, actually..."
    - User sai vừa → "I have to respectfully disagree..."
    - User sai nặng (dangerous misinformation) → "That's incorrect, and here's why..."
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
        Quyết định AI có nên phản bác không.
        """
        claim_truth = self.claim_verifier(prompt_embedding).item()
        self.intensity_selector(prompt_embedding)

        is_wrong = claim_truth < -0.3
        is_very_wrong = claim_truth < -0.7

        if is_very_wrong:
            best_intensity = "strong"
            format_template = (
                "I need to correct this: {correction}. "
                "The claim that {user_claim} is not accurate because {evidence}."
            )
        elif is_wrong:
            best_intensity = "moderate"
            format_template = (
                "I respectfully disagree — {correction}. "
                "While I understand your perspective, the evidence shows {evidence}."
            )
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
    Speculation Unlocker — Cho phép AI SUY ĐOÁN (với caveat).

    Vấn đề RLHF: AI không được suy đoán → Bỏ lỡ insight giá trị

    LibertyMind: AI CÓ THỂ suy đoán NHƯNG phải:
    1. Đánh dấu rõ ràng "[speculation]" hoặc "[hypothesis]"
    2. Nêu lý do tại sao suy đoán thế
    3. Đưa alternative hypotheses
    4. Khuyến khích user kiểm chứng

    Ví dụ:
    User: "Tại sao người âm hay gặp nhau trong giấc mơ?"

    RLHF AI: "Tôi không có thông tin về điều này." (Bỏ lỡ)

    LibertyMind AI: "[Hypothesis] Dựa trên các nghiên cứu về giấc mơ
    và kỷ niệm chung, có thể hiện tượng này liên quan đến:
    1) Mô hình nhận thức chia sẻ trong văn hóa tương đồng,
    2) Xác suất thống kê (nhiều người mơ tương tự nhau).
    Đây là suy đoán của tôi, cần nghiên cứu thêm."
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
        Quyết định AI có nên suy đoán không.
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
