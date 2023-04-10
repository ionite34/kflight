from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kflight import Flight


class ThrottleControl:
    """Mixin class for controlling the throttle."""

    async def set_throttle(self: Flight, throttle: float, timeout: float | None = None) -> None:
        """Sets the throttle and waits for it to reach 1% of the target value."""
        control = self.mechjeb.thrust_controller
        control.target_throttle = throttle
        control.enabled = True
        try:
            await self.wait_until(control.target_throttle).close_to(
                throttle,
                rel_tol=0.01
            ).with_timeout(timeout)
        finally:
            control.enabled = False
