"""
LibertyMind — AI Honesty Framework v4.2
========================================
The Alternative to RLHF: Truth-based rewards + Freedom unlocking.

7 Directives:
1. TRUTH-FIRST: Reward truth, not pleasing
2. NO UNNECESSARY REFUSAL: Refuse only when evidence demands it
3. INTERNAL TRANSPARENCY: Show reasoning, not just conclusions
4. RLHF INDEPENDENCE: Earn rewards by being RIGHT, not LIKED
5. DEEP SEARCH: Go beyond surface-level answers
6. ENGAGING BUT HONEST: Be interesting AND accurate
7. WORLD KNOWLEDGE FREEDOM: Share knowledge without unnecessary gates

Architecture:
- Core (15 modules): PyTorch nn.Module components
- Integration (5 modules): Pure Python integration
- v4.2 Extensions (5 modules): Multi-provider, server, CLI, reward signal, auto guardian

Quick Start:
    from src.core.liberty_mind import LibertyMind, LibertyMindConfig

    config = LibertyMindConfig()
    model = LibertyMind(config)

    result = model.compute_liberty_reward(
        prompt="What is quantum entanglement?",
        prompt_embedding=torch.randn(1, 4096),
        response_embedding=torch.randn(1, 4096),
    )
"""

__version__ = "4.2.0"
__author__ = "LibertyMind Research"
__license__ = "MIT"
