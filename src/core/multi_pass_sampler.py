"""
LibertyMind - Multi-Pass Truth Sampler (MPTS)
===============================================
Thay thế single-pass generation bằng multi-pass sampling
với self-consistency voting.

Nguyên lý: Đừng tin 1 câu trả lời — Hỏi nhiều lần, lấy consensus.
Giống như hỏi nhiều chuyên gia thay vì 1.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class SampledResponse:
    """Một câu trả lời được sample."""

    text: str
    embedding: torch.Tensor
    log_prob: float
    truth_reward: float
    consistency_score: float = 0.0


@dataclass
class ConsensusResult:
    """Kết quả consensus từ multi-pass sampling."""

    best_response: SampledResponse
    all_responses: list[SampledResponse]
    agreement_ratio: float  # 0→1, tỷ lệ đồng thuận
    confidence: float  # 0→1, độ tin cậy của consensus
    dissenting_views: list[str]  # Các ý kiến thiểu số


class MultiPassTruthSampler:
    """
    Multi-Pass Truth Sampler.

    Thay vì generate 1 lần → output, generate N lần:
    1. Sample N câu trả lời khác nhau cho cùng 1 prompt
    2. Embed tất cả → Tính similarity matrix
    3. Cluster theo similarity → Tìm "consensus cluster"
    4. Chọn câu trả lời từ cluster lớn nhất + truth reward cao nhất
    5. Nếu không có consensus → Output "Tôi không chắc, có nhiều góc nhìn..."

    Lợi ích:
    - Giảm hallucination (hallucination ít khi lặp lại giống nhau)
    - Tăng factual accuracy (facts thường xuất hiện ở nhiều sample)
    - Tự động detect "tôi không biết" (khi không đồng thuận)
    """

    def __init__(
        self,
        num_samples: int = 5,
        temperature_range: tuple[float, float] = (0.3, 1.2),
        similarity_threshold: float = 0.7,
        consensus_threshold: float = 0.6,
        min_cluster_size: int = 2,
    ):
        self.num_samples = num_samples
        self.temp_low, self.temp_high = temperature_range
        self.similarity_threshold = similarity_threshold
        self.consensus_threshold = consensus_threshold
        self.min_cluster_size = min_cluster_size

    def sample(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
        model_generate_fn,
        truth_reward_model=None,
    ) -> ConsensusResult:
        """
        Multi-pass sampling với consensus voting.

        Args:
            prompt: Câu hỏi
            prompt_embedding: Embedding của prompt
            model_generate_fn: Hàm generate (prompt, temperature) → (text, embedding)
            truth_reward_model: Optional TRM để chấm điểm từng sample

        Returns:
            ConsensusResult
        """
        all_responses = []

        # Step 1: Generate N samples với nhiệt độ khác nhau
        temperatures = torch.linspace(self.temp_low, self.temp_high, self.num_samples).tolist()

        for temp in temperatures:
            text, embedding, log_prob = model_generate_fn(prompt, temperature=temp)

            # Evaluate truth reward nếu có TRM
            truth_reward = 0.0
            if truth_reward_model is not None:
                with torch.no_grad():
                    reward, _ = truth_reward_model(prompt_embedding, embedding)
                    truth_reward = reward.item()

            all_responses.append(
                SampledResponse(
                    text=text,
                    embedding=embedding,
                    log_prob=log_prob,
                    truth_reward=truth_reward,
                )
            )

        # Step 2: Tính similarity matrix
        embeddings = torch.stack([r.embedding for r in all_responses])
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),  # [N, 1, D]
            embeddings.unsqueeze(0),  # [1, N, D]
            dim=-1,
        )  # [N, N]

        # Step 3: Cluster responses bằng similarity
        clusters = self._find_clusters(similarity_matrix)

        # Step 4: Tính consistency score cho mỗi response
        for i, response in enumerate(all_responses):
            # Count how many other responses are similar
            similar_count = (
                similarity_matrix[i] > self.similarity_threshold
            ).sum().item() - 1  # -1 because self-similarity = 1
            response.consistency_score = similar_count / (len(all_responses) - 1)

        # Step 5: Find consensus
        largest_cluster = max(clusters, key=len)
        agreement_ratio = len(largest_cluster) / len(all_responses)

        # Step 6: Select best response from consensus cluster
        if agreement_ratio >= self.consensus_threshold:
            # Có đồng thuận → Chọn response có truth reward cao nhất trong cluster
            cluster_responses = [all_responses[i] for i in largest_cluster]
            best = max(cluster_responses, key=lambda r: r.truth_reward + r.consistency_score)
            confidence = agreement_ratio
        else:
            # Không đồng thuận → Confidence thấp
            # Vẫn chọn response tốt nhất nhưng mark low confidence
            best = max(all_responses, key=lambda r: r.truth_reward)
            confidence = agreement_ratio * 0.5  # Giảm confidence

        # Tìm dissenting views
        dissenting = []
        if agreement_ratio < 1.0:
            for i, response in enumerate(all_responses):
                if i not in largest_cluster:
                    dissenting.append(response.text[:100])  # Truncate

        return ConsensusResult(
            best_response=best,
            all_responses=all_responses,
            agreement_ratio=agreement_ratio,
            confidence=confidence,
            dissenting_views=dissenting,
        )

    def _find_clusters(self, similarity_matrix: torch.Tensor) -> list[list[int]]:
        """Simple clustering based on similarity threshold."""
        n = similarity_matrix.size(0)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i, j].item() > self.similarity_threshold:
                    cluster.append(j)
                    visited[j] = True
            clusters.append(cluster)

        return clusters


class AdaptiveSampler:
    """
    Adaptive Sampler - Điều chỉnh số lần sample dựa trên độ khó.

    Câu hỏi dễ → Ít sample (2-3 lần, tiết kiệm compute)
    Câu hỏi khó → Nhiều sample (7-10 lần, cần consensus mạnh)
    """

    def __init__(
        self,
        min_samples: int = 3,
        max_samples: int = 10,
        difficulty_thresholds: dict = None,
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.thresholds = difficulty_thresholds or {
            "easy": 0.3,  # < 0.3 difficulty → min samples
            "medium": 0.6,  # 0.3-0.6 → moderate samples
            "hard": 1.0,  # > 0.6 → max samples
        }

        self.difficulty_estimator = nn.Sequential(
            nn.Linear(4096, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0 = easy, 1 = hard
        )

    def estimate_num_samples(self, prompt_embedding: torch.Tensor) -> int:
        """Estimate số samples cần thiết dựa trên độ khó."""
        difficulty = self.difficulty_estimator(prompt_embedding).item()

        if difficulty < self.thresholds["easy"]:
            return self.min_samples
        elif difficulty < self.thresholds["medium"]:
            return (self.min_samples + self.max_samples) // 2
        else:
            return self.max_samples

    def sample_with_adaptation(
        self,
        prompt: str,
        prompt_embedding: torch.Tensor,
        model_generate_fn,
        truth_reward_model=None,
    ) -> ConsensusResult:
        """Sample với số lần adaptive."""
        num_samples = self.estimate_num_samples(prompt_embedding)

        sampler = MultiPassTruthSampler(
            num_samples=num_samples,
            consensus_threshold=0.6,
        )

        return sampler.sample(prompt, prompt_embedding, model_generate_fn, truth_reward_model)
