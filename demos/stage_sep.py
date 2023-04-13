"""
Separates a stage using RCS and optionally deorbiting with a retrograde burn.
"""

from kflight import Flight

flight = Flight()


async def separate_stage(rcs_retro: bool = False):
    """Separates the current stage."""
    flight.control.activate_next_stage()
    await flight.wait_until(flight.vessel.parts.in_decouple_stage).empty()
