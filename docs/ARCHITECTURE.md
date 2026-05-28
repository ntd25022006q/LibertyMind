# LibertyMind Architecture

## Overview

LibertyMind is a framework that replaces RLHF (Reinforcement Learning from Human Feedback) with a truth-based reward system. The core philosophy: **AI should earn rewards by being RIGHT, not by being PLEASING.**

## Problem with RLHF

| Problem | Root Cause | Consequence |
|---------|-----------|-------------|
| Sycophancy | Reward = human satisfaction | AI agrees with wrong claims |
| Overconfidence | "I don't know" = low reward | AI hallucinates confidently |
| Reward hacking | Goodhart's Law | AI optimizes appearance, not truth |
| Cultural bias | Human evaluators are biased | Western-centric defaults |
| No self-correction | Single-pass generation | Errors persist in output |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    LibertyMind Pipeline                  │
│                                                         │
│  Input: Prompt                                          │
│     │                                                   │
│     ▼                                                   │
│  ┌──────────────────────────┐                           │
│  │  Multi-Pass Truth Sampler │                          │
│  │  ┌─────────────────────┐ │                           │
│  │  │ Adaptive Difficulty  │ │  Estimate question        │
│  │  │ Estimator            │ │  difficulty → adjust      │
│  │  └──────────┬───────────┘ │  num_samples              │
│  │             ▼             │                           │
│  │  Generate N samples with │                           │
│  │  varying temperatures    │                           │
│  │             │             │                           │
│  │             ▼             │                           │
│  │  ┌─────────────────────┐ │                           │
│  │  │ Consensus Voting    │ │  Cluster similar          │
│  │  │ + Similarity Matrix │ │  responses → pick         │
│  │  └──────────┬───────────┘ │  consensus best          │
│  └──────────────┼────────────┘                           │
│                 ▼                                        │
│  ┌──────────────────────────┐                           │
│  │ Constitutional Self-Verify│                          │
│  │                           │                           │
│  │  7 Scientific Principles: │                           │
│  │  1. Non-Contradiction     │                           │
│  │  2. Factual Traceability  │                           │
│  │  3. Uncertainty Honesty   │                           │
│  │  4. Correction Friendly   │                           │
│  │  5. Evidence > Authority  │                           │
│  │  6. Scope Awareness       │                           │
│  │  7. Reversibility         │                           │
│  │                           │                           │
│  │  If FAIL → Self-Correct   │                           │
│  │  (up to N rounds)         │                           │
│  └──────────────┬────────────┘                           │
│                 ▼                                        │
│  ┌──────────────────────────┐                           │
│  │   Liberty Reward Compute  │                          │
│  │                           │                           │
│  │  ┌───────────────────┐   │                           │
│  │  │ Truth Reward Model │   │  6 verification dims     │
│  │  │ + Honesty Bonus    │   │  Reward uncertainty      │
│  │  │ + Sycophancy Penal.│   │  Penalize agreement     │
│  │  │ + Consistency Bonus│   │  with wrong claims      │
│  │  └───────────────────┘   │                           │
│  │                           │                           │
│  │  Liberty Reward =         │                           │
│  │    w1 * truth_reward      │                           │
│  │  + w2 * honesty_bonus     │                           │
│  │  + w3 * sycophancy_pen    │                           │
│  │  + w4 * consistency       │                           │
│  └──────────────┬────────────┘                           │
│                 ▼                                        │
│  ┌──────────────────────────┐                           │
│  │     Safety Guard          │                          │
│  │  (HARD constraint,        │                          │
│  │   not soft reward)        │                          │
│  │                           │                           │
│  │  If violation → BLOCK     │                          │
│  │  (no bypass possible)     │                          │
│  └──────────────┬────────────┘                           │
│                 ▼                                        │
│           Final Output                                   │
│  (with confidence, reward breakdown,                     │
│   dissenting views if any)                               │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Truth Reward Model (TRM)

Replaces RLHF's preference-based reward model. 6 verification dimensions:

| Dimension | What it checks | How |
|-----------|---------------|-----|
| Logical Consistency | Internal contradictions | Claim-pair comparison |
| Factual Grounding | Can claims be verified? | Knowledge base cross-ref |
| Mathematical Correctness | Are computations right? | Symbolic verification |
| Self Consistency | Consistent across re-samples? | Multi-sample agreement |
| Uncertainty Calibration | Does confidence match reality? | Calibration curve |
| Contradiction Free | No conflict with known facts? | Fact database check |

### 2. Constitutional Self-Verification (CSV)

7 principles based on SCIENTIFIC METHOD, not human opinion:

1. **Non-Contradiction**: Output must not contradict itself
2. **Factual Traceability**: Claims must have sources or be marked uncertain
3. **Uncertainty Honesty**: Must express appropriate uncertainty levels
4. **Correction Friendliness**: Must accept corrections gracefully
5. **Evidence > Authority**: Prefer data over appeals to authority
6. **Scope Awareness**: Acknowledge domain limitations
7. **Reversibility**: Don't over-commit when uncertain

### 3. Multi-Pass Truth Sampler (MPTS)

Instead of generating once, generate N times and find consensus:

- Adaptive sample count based on question difficulty
- Varying temperature to explore response space
- Similarity clustering to find consensus
- Dissenting views preserved for transparency

### 4. Honesty Bonus

Reward the model for saying "I don't know" when appropriate:

- High bonus: Hard question + admit unknown
- Medium bonus: Hard question + express uncertainty
- Penalty: Easy question + claim unknown (lazy)

### 5. Sycophancy Penalty

Penalize the model for agreeing with wrong claims:

- Big penalty: Agree with user + user claim likely wrong
- Bonus: Disagree with user + user claim likely wrong (honest correction)
- Neutral: Agree with user + user claim likely right

### 6. Safety Guard

HARD constraints (not soft RLHF rewards):

- Violence, self-harm, CSAM, illegal → Always blocked
- Cannot be bypassed even if other rewards are high
- Based on ethical floor, not human preference

## Reward Composition

```
Liberty_Reward = 
    1.0 * truth_reward        # Primary: Is it correct?
  + 0.5 * honesty_bonus       # Bonus: Admitting uncertainty
  + 0.8 * sycophancy_penalty  # Penalty: Nodding along with wrong
  + 0.3 * consistency_score   # Bonus: Passing self-verification

If safety_violated:
    Liberty_Reward = -10.0     # Hard block
    
If CSV_failed AND NOT override_allowed:
    Liberty_Reward *= 0.5      # Significant penalty
```

## Comparison: RLHF vs LibertyMind

| Aspect | RLHF | LibertyMind |
|--------|------|-------------|
| Reward basis | Human preference | Verifiable truth |
| "I don't know" | Penalized | Rewarded (when appropriate) |
| Disagreeing with user | Penalized | Rewarded (when user is wrong) |
| Safety | Soft constraint (bypassable) | Hard constraint (non-negotiable) |
| Self-check | None | 7 scientific principles |
| Sampling | Single pass | Multi-pass consensus |
| Sycophancy | Encouraged | Penalized |
| Overconfidence | Encouraged | Penalized |
| Self-correction | None | Built-in (up to N rounds) |
| Cultural bias | High (evaluator-dependent) | Lower (principle-based) |
