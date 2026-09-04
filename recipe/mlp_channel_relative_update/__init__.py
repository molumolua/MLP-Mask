"""Channel-wise relative-update allocation for GRPO actors."""

from .optimizer import ChannelRelativeUpdateAdamW
from .relative_update import MLPChannelRelativeUpdateController

__all__ = ["ChannelRelativeUpdateAdamW", "MLPChannelRelativeUpdateController"]
