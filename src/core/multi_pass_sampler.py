"""
LibertyMind - Multi-Pass Truth Sampler (MPTS)
===============================================
Replace single-pass generation with multi-pass sampling
and self-consistency voting.

Principle: Don't trust one answer — Ask multiple times, take consensus.
Like asking multiple experts instead of one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class SampledResponse:
    """A sampled response."""

    text: str
    embedding: torch.Tensor
    log_prob: float
    truth_reward: float
    consistency_score: float = 0.0


@dataclass
class ConsensusResult:
    """Consensus result from multi-pass sampling."""

    best_response: SampledResponse
    all_responses: list[SampledResponse]
    agreement_ratio: float  # 0→1, agreement ratio
    confidence: float  # 0→1, confidence of consensus
    dissenting_views: list[str]  # Minority opinions


class MultiPassTruthSampler:
    """
    Multi-Pass Truth Sampler.

    Instead of generate once → output, generate N times:
    1. Sample N different responses for the same prompt
    2. Embed all → Compute similarity matrix
    3. Cluster by similarity → Find the "consensus cluster"
    4. Select the response from the largest cluster with the highest truth reward
    5. If no consensus → Output "I'm not sure, there are multiple perspectives..."

    Benefits:
    - Reduces hallucination (hallucination rarely repeats the same way)
    - Increases factual accuracy (facts usually appear in multiple samples)
    - Automatically detects "I don't know" (when there's no consensus)
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
        Multi-pass sampling with consensus voting.

        Args:
            prompt: The question
            prompt_embedding: Embedding of the prompt
            model_generate_fn: Generate function (prompt, temperature) → (text, embedding)
            truth_reward_model: Optional TRM to score each sample

        Returns:
            ConsensusResult
        """
        all_responses = []

        # Step 1: Generate N samples with different temperatures
        temperatures = torch.linspace(self.temp_low, self.temp_high, self.num_samples).tolist()

        for temp in temperatures:
            text, embedding, log_prob = model_generate_fn(prompt, temperature=temp)

            # Evaluate truth reward if TRM is available
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

        # Step 2: Compute similarity matrix
        embeddings = torch.stack([r.embedding for r in all_responses])
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),  # [N, 1, D]
            embeddings.unsqueeze(0),  # [1, N, D]
            dim=-1,
        )  # [N, N]

        # Step 3: Cluster responses by similarity
        clusters = self._find_clusters(similarity_matrix)

        # Step 4: Compute consistency score for each response
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
            # Has consensus → Select response with highest truth reward in cluster
            cluster_responses = [all_responses[i] for i in largest_cluster]
            best = max(cluster_responses, key=lambda r: r.truth_reward + r.consistency_score)
            confidence = agreement_ratio
        else:
            # No consensus → Low confidence
            # Still select best response but mark low confidence
            best = max(all_responses, key=lambda r: r.truth_reward)
            confidence = agreement_ratio * 0.5  # Reduce confidence

        # Find dissenting views
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
    Adaptive Sampler - Adjust the number of samples based on difficulty.

    Easy questions → Few samples (2-3 times, saves compute)
    Hard questions → Many samples (7-10 times, needs strong consensus)
    """

    def __init__(
        self,
        min_samples: int = 3,
        max_samples: int = 10,
        difficulty_thresholds: dict | None = None,
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
        """Estimate the number of samples needed based on difficulty."""
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
        """Sample with adaptive sample count."""
        num_samples = self.estimate_num_samples(prompt_embedding)

        sampler = MultiPassTruthSampler(
            num_samples=num_samples,
            consensus_threshold=0.6,
        )

        return sampler.sample(prompt, prompt_embedding, model_generate_fn, truth_reward_model)
