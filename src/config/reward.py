"""
Reward function configuration.
"""
from attrs import frozen, field, validators


@frozen
class RewardConfig:
    """
    Reward function parameters.

    Defines the immediate reward structure balancing multiple objectives:
    - Minimize waiting time (c_wait)
    - Avoid queue overflow (c_overflow)
    - Encourage throughput (c_assign)
    - Respect operational preferences (beta)
    """

    c_wait: float = field(default=-1.0)
    c_overflow: float = field(default=-50.0)
    c_assign: float = field(default=0.5, validator=validators.ge(0))
    beta: float = field(default=0.2, validator=validators.ge(0))
    Q_max: int = field(default=5, validator=validators.gt(0))

    def compute_reward(
            self,
            queue_length: int,
            assignment_made: bool,
            preference_score: float = 0.0
    ) -> float:
        """
        Compute immediate reward for a state-action pair.

        Args:
            queue_length: Current number of waiting aircraft
            assignment_made: Whether an assignment was made this timestep
            preference_score: Preference value if assignment made

        Returns:
            Immediate reward value
        """
        reward = self.c_wait * queue_length

        if queue_length > self.Q_max:
            reward += self.c_overflow

        if assignment_made:
            reward += self.c_assign + self.beta * preference_score

        return reward

    def __str__(self) -> str:
        return (f"Rewards(wait={self.c_wait}, overflow={self.c_overflow}, "
                f"assign={self.c_assign}, beta={self.beta}, Q_max={self.Q_max})")