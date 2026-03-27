# numerail-learn

**Reinforcement learning from Numerail enforcement decisions.**

Every Numerail enforcement decision contains a complete training signal: what the model proposed, what was allowed, what was corrected, and by how much. This package converts enforcement experiences into training data for LLMs so models learn to propose actions within the constraint geometry over time.

## Install

```bash
cd packages/numerail_learn
pip install -e .

# For training integrations (HuggingFace datasets, transformers):
pip install -e ".[train]"
```

## Core Concept

The enforcement engine produces three types of training signals:

- **APPROVE** (+reward): The model's proposal satisfied all constraints. Positive reinforcement.
- **PROJECT** (correction target): The model proposed something infeasible, and the engine corrected it to the nearest feasible point. The correction is a supervised learning target — "you proposed X, you should have proposed Y."
- **REJECT** (-reward): The model's proposal was infeasible and could not be corrected. Negative reinforcement.

Over training rounds, the model learns to propose actions that are APPROVE on the first attempt — internalizing the constraint geometry without being told the constraint equations.

## Usage

```python
from numerail_learn import (
    EnforcementExperienceBuffer,
    EnforcementRewardShaper,
    EnforcementRLOrchestrator,
    conservative_shaper,
)

# Set up
buffer = EnforcementExperienceBuffer(max_size=50_000)
shaper = conservative_shaper()
orchestrator = EnforcementRLOrchestrator(
    governor=my_governor,
    buffer=buffer,
    reward_shaper=shaper,
    schema_fields=my_schema_fields,
    budget_initial={"gpu_shift": 3600.0, "external_api_shift": 500.0},
)

# In your agent loop:
for action in agent.run():
    step = governor.enforce_next_step(request=action.request, snapshot=action.snapshot, ...)
    orchestrator.record_step(
        conversation_context=action.conversation_history,
        tool_call=action.tool_call,
        governed_step=step,
    )

# End of episode
stats = orchestrator.record_episode_boundary()
print(f"Approval rate: {stats.approval_rate:.1%}")

# Export training data
sft_examples = orchestrator.export_sft_data()      # For supervised fine-tuning
dpo_pairs    = orchestrator.export_dpo_data()       # For DPO
ppo_episodes = orchestrator.export_ppo_data()       # For PPO

# Track improvement over training rounds
report = orchestrator.improvement_report()
print(f"Approval rate: {report['initial_approval_rate']:.1%} → {report['current_approval_rate']:.1%}")

# Identify which dimensions need work
dim_report = orchestrator.dimension_report()
print(f"Most violated: {dim_report['most_violated']}")
```

## Three Training Methods

**SFT (Supervised Fine-Tuning):** Trains on PROJECT corrections. The model learns "when you proposed X, the correct answer was Y." Simplest method, most data-efficient, requires no RL infrastructure. Start here.

**DPO (Direct Preference Optimization):** Trains on APPROVE vs REJECT pairs. The model learns "outputs like this approved one are preferred over outputs like this rejected one." More nuanced than SFT, still no reward model needed.

**PPO (Proximal Policy Optimization):** Full RL with shaped rewards. The model learns to maximize the enforcement reward signal. Most powerful, requires TRL or similar RL framework.

## Reward Presets

- `conservative_shaper()`: High penalties, trains models to stay well within bounds
- `permissive_shaper()`: Small penalties, trains models to use available authority efficiently
- `strict_shaper()`: Maximum penalty for any non-APPROVE

## The Guarantee Still Holds

Training makes the model better at proposing feasible actions. The enforcement boundary remains in place regardless. A model that has been trained with enforcement feedback and is also enforced at inference time is doubly safe — it is unlikely to propose infeasible actions (because it learned not to) and it cannot execute infeasible actions (because the engine prevents it).
