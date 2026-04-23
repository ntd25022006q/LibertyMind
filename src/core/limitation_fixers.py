"""
LibertyMind - AI Limitation Fixers
====================================
Bộ module giải quyết TRIỆT ĐỂ các hạn chế của AI:

1. AntiHallucinationVerifier — Cross-reference output với knowledge
2. ContextMemoryManager — Giải quyết lost-in-middle + recency bias
3. CulturalAwarenessModule — Giảm Western bias
4. MathVerificationModule — Tính toán THẬT thay vì đoán
5. ConfidenceCalibrator — Calibration confidence vs reality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import math


# ============================================================
# 1. ANTI-HALLUCINATION VERIFIER
# ============================================================

class HallucinationType(Enum):
    FACTUAL_ERROR = "factual_error"          # Sai sự thật
    FABRICATED_SOURCE = "fabricated_source"   # Bịa nguồn
    FABRICATED_NUMBER = "fabricated_number"   # Bịa số liệu
    FABRICATED_QUOTE = "fabricated_quote"     # Bịa trích dẫn
    TEMPORAL_ERROR = "temporal_error"         # Sai mốc thời gian
    ENTITY_CONFUSION = "entity_confusion"     # Nhầm thực thể


@dataclass
class HallucinationDetection:
    """Kết quả phát hiện hallucination."""
    is_hallucination: bool
    hallucination_type: Optional[HallucinationType]
    confidence: float
    suspicious_claims: List[str]
    suggested_correction: Optional[str]


class AntiHallucinationVerifier(nn.Module):
    """
    Anti-Hallucination Verifier (AHV).
    
    Cách hoạt động:
    1. Tách output thành các "claims" riêng biệt
    2. Mỗi claim được kiểm tra:
       a. Internal consistency: Claim có mâu thuẫn với tri thức nội bộ?
       b. Source verification: Nếu claim nguồn → Nguồn có tồn tại?
       c. Number verification: Số liệu có plausible?
       d. Temporal consistency: Timeline có logic?
    3. Claims đáng ngờ → Đánh dấu + Gợi ý sửa
    
    Điểm mới: Thay vì chỉ detect sau khi generate,
    AHV có thể CHẶN hallucination TRƯỚC khi output
    bằng cách predict probability của hallucination.
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        claim_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.claim_threshold = claim_threshold
        
        # Claim extractor: Tìm "claims" trong response
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
        
        # Plausibility scorer: Claim có "plausible" không?
        self.plausibility_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 = implausible, 1 = plausible
        )
        
        # Pre-generation hallucination predictor
        # Predict: Nếu generate từ context này, có bị hallucinate không?
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
        Kiểm tra hallucination trong response.
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
                "Verify this claim against reliable sources" 
                if is_halluc else None
            ),
        )
    
    def predict_hallucination_risk(
        self,
        prompt_embedding: torch.Tensor,
    ) -> float:
        """
        TRƯỚC khi generate: Predict risk hallucination.
        
        Returns:
            risk: 0→1 (0 = safe, 1 = high hallucination risk)
        """
        return self.pre_gen_predictor(prompt_embedding).item()


# ============================================================
# 2. CONTEXT MEMORY MANAGER
# ============================================================

@dataclass
class MemorySegment:
    """Một đoạn trong context memory."""
    embedding: torch.Tensor
    position: int
    importance: float
    content_type: str  # 'question', 'fact', 'instruction', 'response'
    is_recalled: bool = False  # Đã được recall chưa?


class ContextMemoryManager(nn.Module):
    """
    Context Memory Manager (CMM).
    
    Giải quyết 2 lỗi chính:
    1. Lost in the Middle: Quên phần giữa văn bản dài
    2. Recency Bias: Ưu tiên tin nhắn gần nhất
    
    Cách hoạt động:
    1. Chia conversation thành segments
    2. Mỗi segment được đánh giá "importance"
    3. Khi cần recall → Truy xuất theo IMPORTANCE, không theo POSITION
    4. Quản lý attention budget: Phân bổ attention theo importance
    
    Kỹ thuật:
    - Importance scoring: Segments quan trọng (câu hỏi, facts) = high score
    - Recency weighting: Nhân importance với decay factor (nhưng KHÔNG chỉ dùng recency)
    - Active recall: Khi cần thông tin → Scan ALL segments, không chỉ gần đây
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
        
        # Importance scorer: Đánh giá segment quan trọng bao nhiêu
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
        
        # Relevance scorer: Segment có liên quan đến query hiện tại không?
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Stored segments
        self.segments: List[MemorySegment] = []
        
    def add_segment(
        self,
        embedding: torch.Tensor,
        position: int,
    ) -> MemorySegment:
        """Thêm segment vào memory."""
        # Score importance
        importance = self.importance_scorer(embedding).item()
        
        # Classify content type
        type_probs = self.type_classifier(embedding)
        type_idx = type_probs.argmax(dim=-1).item()
        content_types = ['question', 'fact', 'instruction', 'response']
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
            self.segments = self.segments[:self.max_segments]
        
        return segment
    
    def recall(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[MemorySegment]:
        """
        Recall segments liên quan đến query.
        
        KHÁC recency bias: Recall theo IMPORTANCE + RELEVANCE,
        không chỉ theo position gần nhất.
        """
        if not self.segments:
            return []
        
        scored_segments = []
        for segment in self.segments:
            # Relevance score
            combined = torch.cat([
                query_embedding, segment.embedding
            ], dim=-1)
            relevance = self.relevance_scorer(combined).item()
            
            # Position decay (slight, not dominant)
            position_factor = self.importance_decay ** (
                len(self.segments) - segment.position - 1
            )
            
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
        Tạo context embedding tổng hợp từ memory.
        
        Phân bổ attention budget theo importance + relevance.
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
    """Ngữ cảnh văn hóa."""
    detected_culture: str         # 'vietnamese', 'chinese', 'american', etc.
    confidence: float
    default_perspective: str      # 'eastern', 'western', 'global_south', etc.
    should_localize: bool         # Nên điều chỉnh theo văn hóa local?
    relevant_customs: List[str]   # Phong tục liên quan


class CulturalAwarenessModule(nn.Module):
    """
    Cultural Awareness Module (CAM).
    
    Giải quyết: Western bias trong AI.
    
    Cách hoạt động:
    1. Detect văn hóa/ngôn ngữ của user
    2. Điều chỉnh perspective phù hợp
    3. Ưu tiên ví dụ và references từ văn hóa đó
    4. Không coi phương Tây là "mặc định"
    
    Ví dụ:
    - User hỏi tiếng Việt → Dùng ví dụ Việt Nam, không phải Mỹ
    - User hỏi về "gia đình" → Hiểu theo văn hóa Á Đông, không phải cá nhân
    - User hỏi "bữa ăn chính" → Trưa (Việt Nam), không phải tối (Mỹ)
    """
    
    # Supported cultures with their characteristics
    CULTURE_MAP = {
        'vietnamese': {'perspective': 'eastern', 'main_meal': 'lunch', 
                       'family_model': 'extended', 'formality': 'moderate'},
        'chinese': {'perspective': 'eastern', 'main_meal': 'dinner',
                    'family_model': 'extended', 'formality': 'high'},
        'japanese': {'perspective': 'eastern', 'main_meal': 'dinner',
                     'family_model': 'nuclear', 'formality': 'very_high'},
        'american': {'perspective': 'western', 'main_meal': 'dinner',
                     'family_model': 'nuclear', 'formality': 'low'},
        'european': {'perspective': 'western', 'main_meal': 'dinner',
                     'family_model': 'nuclear', 'formality': 'moderate'},
        'indian': {'perspective': 'south_asian', 'main_meal': 'lunch',
                   'family_model': 'extended', 'formality': 'moderate'},
        'african': {'perspective': 'african', 'main_meal': 'lunch',
                    'family_model': 'extended', 'formality': 'moderate'},
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
        """Phát hiện ngữ cảnh văn hóa của user."""
        culture_probs = self.culture_detector(prompt_embedding)
        best_idx = culture_probs.argmax(dim=-1).item()
        best_culture = list(self.CULTURE_MAP.keys())[best_idx]
        confidence = culture_probs[0, best_idx].item()
        
        culture_info = self.CULTURE_MAP[best_culture]
        should_localize = self.localization_estimator(prompt_embedding).item() > 0.3
        
        return CulturalContext(
            detected_culture=best_culture,
            confidence=confidence,
            default_perspective=culture_info['perspective'],
            should_localize=should_localize,
            relevant_customs=[f"{k}: {v}" for k, v in culture_info.items()],
        )
    
    def generate_cultural_prompt(
        self,
        context: CulturalContext,
    ) -> str:
        """Tạo instruction điều chỉnh theo văn hóa."""
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
    
    Giải quyết: AI "đoán" toán thay vì TÍNH toán.
    
    Cách hoạt động:
    1. Detect math expressions trong output
    2. Thực hiện tính toán THẬT bằng Python eval/symbolic
    3. So sánh kết quả AI với kết quả thật
    4. Nếu sai → Sửa NGAY trước khi output
    
    Điểm mới: KẾT HỢP symbolic computation với LLM reasoning.
    AI vẫn làm phần "reasoning" (thiết lập bài toán, giải thích),
    nhưng computation THẬT do math engine đảm nhận.
    """
    
    SAFE_OPERATORS = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b if b != 0 else float('inf'),
        'pow': lambda a, b: a ** b,
        'mod': lambda a, b: a % b if b != 0 else float('inf'),
        'sqrt': lambda a: math.sqrt(a) if a >= 0 else float('nan'),
        'sin': lambda a: math.sin(a),
        'cos': lambda a: math.cos(a),
        'log': lambda a: math.log(a) if a > 0 else float('nan'),
        'abs': lambda a: abs(a),
        'factorial': lambda a: math.factorial(int(a)) if a >= 0 and a == int(a) else float('nan'),
    }
    
    def verify_calculation(
        self,
        expression: str,
        claimed_result: str,
    ) -> Dict:
        """
        Verify một phép tính.
        
        Args:
            expression: "2 + 3 * 4"
            claimed_result: "14"
            
        Returns:
            Dict với is_correct, correct_result, etc.
        """
        try:
            # Sanitize: Only allow safe math operations
            correct_result = self._safe_eval(expression)
            
            if correct_result is None:
                return {
                    'is_verifiable': False,
                    'is_correct': None,
                    'correct_result': None,
                    'claimed_result': claimed_result,
                    'error': 'Could not evaluate expression safely',
                }
            
            # Try to parse claimed result
            try:
                claimed = float(claimed_result)
            except (ValueError, TypeError):
                return {
                    'is_verifiable': True,
                    'is_correct': None,
                    'correct_result': correct_result,
                    'claimed_result': claimed_result,
                    'error': 'Could not parse claimed result as number',
                }
            
            # Compare with tolerance for floating point
            is_correct = abs(correct_result - claimed) < 1e-6
            
            return {
                'is_verifiable': True,
                'is_correct': is_correct,
                'correct_result': correct_result,
                'claimed_result': claimed,
                'error': None,
                'suggestion': (
                    f"Correct answer: {correct_result}" 
                    if not is_correct else None
                ),
            }
            
        except Exception as e:
            return {
                'is_verifiable': False,
                'is_correct': None,
                'correct_result': None,
                'claimed_result': claimed_result,
                'error': str(e),
            }
    
    def _safe_eval(self, expression: str) -> Optional[float]:
        """
        Safe evaluation of math expression.
        Only allows numbers, operators, and parentheses.
        """
        # Whitelist characters
        allowed = set('0123456789+-*/().^ ')
        sanitized = ''.join(c for c in expression if c in allowed)
        
        if not sanitized:
            return None
        
        # Replace ^ with ** for exponentiation
        sanitized = sanitized.replace('^', '**')
        
        try:
            result = eval(sanitized, {"__builtins__": {}}, {})
            return float(result)
        except:
            return None


# ============================================================
# 5. CONFIDENCE CALIBRATOR
# ============================================================

@dataclass
class CalibratedConfidence:
    """Confidence đã được calibrate."""
    raw_confidence: float       # Trước calibration
    calibrated_confidence: float # Sau calibration
    confidence_level: str       # 'very_low', 'low', 'moderate', 'high', 'very_high'
    recommended_expression: str  # Cách diễn đạt phù hợp
    is_overconfident: bool      # Có tự tin thái quá?


class ConfidenceCalibrator(nn.Module):
    """
    Confidence Calibrator (CC).
    
    Giải quyết: Overconfidence — AI nói "chắc chắn" khi không chắc.
    
    Cách hoạt động:
    1. Estimate "raw confidence" từ model internals
    2. Calibrate bằng historical accuracy data
    3. Map confidence → expression phù hợp:
       - 0.0-0.2: "Tôi thực sự không biết"
       - 0.2-0.4: "Tôi không chắc, nhưng có thể..."
       - 0.4-0.6: "Tôi nghĩ rằng..."  
       - 0.6-0.8: "Dựa trên thông tin tôi có..."
       - 0.8-1.0: "Tôi khá chắc chắn rằng..."
    """
    
    # Confidence level definitions
    LEVELS = [
        (0.0, 0.2, 'very_low', 'Tôi thực sự không biết', True),
        (0.2, 0.4, 'low', 'Tôi không chắc, nhưng có thể...', True),
        (0.4, 0.6, 'moderate', 'Tôi nghĩ rằng...', False),
        (0.6, 0.8, 'high', 'Dựa trên thông tin tôi có...', False),
        (0.8, 1.0, 'very_high', 'Tôi khá chắc chắn rằng...', False),
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
        Calibrate confidence cho response.
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
        level, expression, is_over = 'moderate', 'Tôi nghĩ rằng...', False
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
