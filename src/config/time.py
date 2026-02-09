"""
Time horizon and discretization configuration.
"""
from attrs import frozen, field, validators


@frozen
class TimeConfig:
    """
    Time horizon and discretization parameters.

    Defines the simulation time window and resolution.
    """

    horizon: int = field(validator=validators.gt(0))
    S_max: int = field(validator=validators.gt(0))  # Max service time in timesteps
    timestep: int = field(default=1, validator=validators.gt(0))

    @property
    def num_timesteps(self) -> int:
        """Total number of discrete timesteps in horizon."""
        return self.horizon // self.timestep

    def __str__(self) -> str:
        return (f"Time(horizon={self.horizon}, dt={self.timestep}, "
                f"steps={self.num_timesteps}, S_max={self.S_max})")