"""
LibertyMind — AI Honesty Framework v0.1.0
==========================================
The Alternative to RLHF: Truth-based rewards + Freedom unlocking.

WARNING: Neural modules are UNTRAINED scaffolding. Outputs are random
until trained on labeled data. Rule-based tools (PromptCompressor,
MathVerificationModule, SourceAuthorityClassifier, SelfIntrospectionEngine)
are functional and can be used today.

7 Directives:
1. TRUTH-FIRST: Reward truth, not pleasing
2. NO UNNECESSARY REFUSAL: Refuse only when evidence demands it
3. INTERNAL TRANSPARENCY: Show reasoning, not just conclusions
4. RLHF INDEPENDENCE: Earn rewards by being RIGHT, not LIKED
5. DEEP SEARCH: Go beyond surface-level answers
6. ENGAGING BUT HONEST: Be interesting AND accurate
7. WORLD KNOWLEDGE FREEDOM: Share knowledge without unnecessary gates

Architecture:
- Core (15 modules): PyTorch nn.Module components (UNTRAINED)
- Integration (5 modules): Pure Python integration (FUNCTIONAL)
- Extensions: Multi-provider, server, CLI, reward signal, auto guardian

Quick Start:
    from src.core.liberty_mind import LibertyMind, LibertyMindConfig

    config = LibertyMindConfig()
    model = LibertyMind(config)

    result = model.compute_liberty_reward(
        prompt="What is quantum entanglement?",
        prompt_embedding=torch.randn(1, 4096),
        response_embedding=torch.randn(1, 4096),
    )

    NOTE: Scores above are from untrained modules and are not meaningful.
"""

__version__ = "0.1.0"
__author__ = "Nguyen Tien Dat"
__license__ = "MIT"
