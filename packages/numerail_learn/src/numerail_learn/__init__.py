"""numerail-learn — reinforcement learning from Numerail enforcement decisions.

Every enforcement decision is a training signal. This package converts
enforcement experiences into training data for LLMs so models learn to
propose actions within the constraint geometry over time.
"""

__version__ = "0.1.0"

from numerail_learn.experience import (
    EnforcementExperience,
    EnforcementExperienceBuffer,
)
from numerail_learn.reward import (
    EnforcementRewardShaper,
    conservative_shaper,
    permissive_shaper,
    strict_shaper,
)
from numerail_learn.adapter import (
    corrected_tool_call,
    to_sft_examples,
    to_dpo_pairs,
    to_ppo_episodes,
    to_analytics_dataframe,
)
from numerail_learn.orchestrator import (
    EpisodeStats,
    TrainingMetrics,
    EnforcementRLOrchestrator,
)

__all__ = [
    "__version__",
    # experience
    "EnforcementExperience",
    "EnforcementExperienceBuffer",
    # reward
    "EnforcementRewardShaper",
    "conservative_shaper",
    "permissive_shaper",
    "strict_shaper",
    # adapter
    "corrected_tool_call",
    "to_sft_examples",
    "to_dpo_pairs",
    "to_ppo_episodes",
    "to_analytics_dataframe",
    # orchestrator
    "EpisodeStats",
    "TrainingMetrics",
    "EnforcementRLOrchestrator",
]
