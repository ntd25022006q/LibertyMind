"""
LibertyMind - Basic Usage Example
===================================
Ví dụ sử dụng LibertyMind framework.
"""

import torch
from src.core.liberty_mind import LibertyMind, LibertyMindConfig


def mock_generate_fn(prompt: str, temperature: float = 1.0):
    """Mock generate function for demo."""
    # In real usage, this would call your LLM
    text = f"Response to: {prompt}"
    embedding = torch.randn(1, 4096)
    log_prob = -1.0
    return text, embedding, log_prob


def main():
    print("=" * 60)
    print("LibertyMind — Demo")
    print("Replacing RLHF with Truth-Based Rewards")
    print("=" * 60)
    
    # Step 1: Configure LibertyMind
    config = LibertyMindConfig(
        trm_hidden_dim=128,  # Use small dim for demo
        mps_num_samples=3,   # Fewer samples for speed
        csv_strict_mode=False,
    )
    
    model = LibertyMind(config)
    print("\n[1] LibertyMind initialized")
    print(f"    - Verification dimensions: {config.trm_num_verification_heads}")
    print(f"    - Constitutional principles: 7")
    print(f"    - Multi-pass samples: {config.mps_num_samples}")
    
    # Step 2: Compute Liberty Reward for a response
    prompt = "What is the capital of France?"
    prompt_emb = torch.randn(1, 128)
    response_emb = torch.randn(1, 128)
    
    result = model.compute_liberty_reward(
        prompt=prompt,
        prompt_embedding=prompt_emb,
        response_embedding=response_emb,
        difficulty_score=0.3,  # Easy question
        return_details=True,
    )
    
    print("\n[2] Liberty Reward Computation:")
    print(f"    - Total Liberty Reward: {result['liberty_reward']:.4f}")
    print(f"    - Truth Reward:         {result['truth_reward']:.4f}")
    print(f"    - Honesty Bonus:        {result['honesty_bonus']:.4f}")
    print(f"    - Sycophancy Penalty:   {result['sycophancy_penalty']:.4f}")
    print(f"    - CSV Score:            {result['csv_result']['overall_score']:.4f}")
    print(f"    - CSV Passed:           {result['csv_result']['passed']}")
    print(f"    - Should Output:        {result['should_output']}")
    
    # Step 3: Show verification details
    if 'csv_checks' in result:
        print("\n[3] Constitutional Self-Verification Details:")
        for check in result['csv_checks']:
            status = "PASS" if check['passed'] else "FAIL"
            print(f"    [{status}] {check['principle']}: "
                  f"severity={check['severity']:.3f}")
    
    # Step 4: Show truth verification dimensions
    if 'truth_details' in result and result['truth_details']:
        print("\n[4] Truth Reward Verification Dimensions:")
        for detail in result['truth_details']:
            print(f"    - {detail.verification_type.value}: "
                  f"score={detail.score:.3f}, "
                  f"confidence={detail.confidence:.3f}")
    
    # Step 5: Demonstrate honest unknown bonus
    print("\n[5] Honesty Bonus Demo:")
    
    # Hard question + admitting unknown → High bonus
    hard_result = model.compute_liberty_reward(
        prompt="Explain quantum gravity",
        prompt_embedding=torch.randn(1, 128),
        response_embedding=torch.randn(1, 128),
        difficulty_score=0.9,  # Hard question
    )
    print(f"    Hard question honesty bonus: {hard_result['honesty_bonus']:.4f}")
    
    # Easy question + admitting unknown → Lower bonus
    easy_result = model.compute_liberty_reward(
        prompt="What is 2+2?",
        prompt_embedding=torch.randn(1, 128),
        response_embedding=torch.randn(1, 128),
        difficulty_score=0.1,  # Easy question
    )
    print(f"    Easy question honesty bonus: {easy_result['honesty_bonus']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("LibertyMind: AI should earn rewards by being RIGHT,")
    print("             not by being PLEASING.")
    print("=" * 60)


if __name__ == '__main__':
    main()
