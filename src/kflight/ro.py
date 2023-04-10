"""Info functions for Realism Overhaul mod."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

from krpc.services.spacecenter import Engine, Part, Module
from pydantic import BaseModel, Field


class ROEngineStatus(StrEnum):
    """Status of a RO (RealFuels) engine."""

    NOMINAL = "Nominal"
    VAPOR_IN_FEED_LINE = "Vapor in feed line"
    NO_FUEL = "No propellants"
    NO_PRESSURE = "Lack of pressure"
    UNDERWATER = "Underwater"
    NO_AIRFLOW = "Airflow outside specs"


class ROEngineInfo(BaseModel):
    """Info about a RO (RealFuels) engine."""

    propellant_stability: float = Field(alias="Propellant")
    predicted_residuals: float = Field(alias="Predicted Residuals")
    mixture_ratio: float = Field(alias="Mixture Ratio")
    ignitions_remaining: int = Field(alias="Ignitions Remaining")
    spool_up_time: float = Field(alias="Effective Spool-Up Time")
    throttle: float = Field(alias="Current Throttle")
    mass_flow: float = Field(alias="Mass Flow")
    internal_temperature: float = Field(alias="Eng. Internal Temp")
    thrust: float = Field(alias="Thrust")
    isp: float = Field(alias="Specific Impulse")
    status: ROEngineStatus = Field(alias="Status")
    supports_throttle: bool = Field(alias="Throttle")

    @classmethod
    def from_engine(cls, engine: Engine | Part) -> ROEngineInfo:
        """Create a new instance from an engine."""
        if isinstance(engine, Engine):
            engine = engine.part

        mod = next(m for m in engine.modules if m.name == "ModuleEnginesRF")
        return cls.from_module(mod)

    @classmethod
    def from_module(cls, module: Module) -> ROEngineInfo:
        """Create a new instance from the `ModuleEnginesRF` module."""
        data: dict[str, str | float] = module.fields.copy()
        propellant_n = re.search(r"(\d.+)\s?%", data["Propellant"])
        if propellant_n is None:
            raise ValueError(f"Could not parse propellant: {data['Propellant']!r}")
        data["Propellant"] = float(propellant_n.group(1))
        return cls(**data)


