from __future__ import annotations

import asyncio
import inspect
import math
from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Awaitable, Callable
from copy import copy
from functools import partial

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Generator, TypeVar, SupportsFloat, SupportsIndex, Protocol, runtime_checkable

from krpc import Client
from krpc.stream import Stream
from typing_extensions import Self
from varname import argname

if TYPE_CHECKING:
    from kflight import Flight


@runtime_checkable
class Comparable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self: CT, other: CT) -> bool:
        pass


T = TypeVar("T")
CT = TypeVar("CT", bound=Comparable)


@dataclass(frozen=True, kw_only=True)
class BaseCondition(ABC, Awaitable[T]):
    conn: Client = field(repr=False)
    inverse: bool = False
    timeout: float | None = None
    predicate: Callable[[T], bool] = field(default=bool, repr=False)

    def _with_kwargs(self, **kwargs) -> Self:
        c = copy(self)
        for name, value in kwargs.items():
            object.__setattr__(c, name, value)  # needed for frozen dataclass
        return c

    def _stream_future(self, stream: Stream) -> asyncio.Future[T]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def callback(value) -> None:
            try:
                pred_match = (
                    self.predicate(value)
                    if not self.inverse
                    else not self.predicate(value)
                )
                if pred_match:
                    stream.remove_callback(callback)
                    # Set the future result to the value
                    loop.call_soon_threadsafe(future.set_result, value)
            except Exception as exc:
                stream.remove_callback(callback)
                loop.call_soon_threadsafe(future.set_exception, exc)
                # Cancel
                raise

        stream.add_callback(callback)
        stream.start(wait=False)

        return future

    def with_timeout(self, timeout: float) -> Self:
        """Returns a new condition that will time out after the given time."""
        return self._with_kwargs(timeout=timeout)

    def equals(self, value: T) -> Self:
        """Returns a new condition that will wait until the property equals the given value."""
        return self._with_kwargs(predicate=lambda x: x == value)

    def close_to(
        self,
        value: SupportsFloat | SupportsIndex,
        rel_tol: float = 1e-09,
        abs_tol: float = 0.0,
    ) -> Self:
        """
        Returns a new condition that will wait until the property
        is within the given tolerance of the value.
        """
        func = partial(math.isclose, value, rel_tol=rel_tol, abs_tol=abs_tol)
        return self._with_kwargs(predicate=func)

    def less_than(self, value: CT) -> Self:
        """Returns a new condition that will wait until the property is less than the given value."""
        return self._with_kwargs(predicate=lambda x: x < value)

    def greater_than(self, value: CT) -> Self:
        """Returns a new condition that will wait until the property is greater than the given value."""
        return self._with_kwargs(predicate=lambda x: x > value)

    def on_predicate(self, predicate: Callable[[T], bool]) -> Self:
        """Returns a new condition that will wait until the predicate returns true."""
        return self._with_kwargs(predicate=predicate)


@dataclass(frozen=True, kw_only=True)
class PropertyCondition(BaseCondition[T]):
    property_of: Any
    property_name: str

    # noinspection PyUnusedLocal
    @classmethod
    def from_statement(cls, conn: Client, remote_property: T, frame: int = 1) -> PropertyCondition[T]:
        """
        Parses the caller's `remote_property` statement and returns a new condition.
        """
        caller_statement = argname(
            "remote_property",
            frame=frame,
            ignore="Flight.wait_until",
            vars_only=False,
        )
        orig_caller_statement = caller_statement

        inverse = False
        if caller_statement.startswith("not "):
            inverse = True
            caller_statement = caller_statement[4:]

        # Validators
        if "." not in caller_statement:
            raise ValueError(f"Attribute access not found in {caller_statement!r}")
        if " " in caller_statement:
            raise ValueError(f"Spaces not allowed in {caller_statement!r}")

        caller_frame_info = inspect.stack()[frame]
        obj_name, property_name = caller_statement.rsplit(".", 1)
        caller_obj = eval(
            obj_name,
            caller_frame_info.frame.f_globals,
            caller_frame_info.frame.f_locals,
        )

        return cls(
            conn=conn,
            property_of=caller_obj,
            property_name=caller_statement.rsplit(".", 1)[1],
            inverse=inverse,
        )

    def __await__(self) -> Generator[T, None, Any]:
        """Defaults to waiting for the property to be truthy."""
        stream = self.conn.add_stream(
            getattr, self.property_of, self.property_name
        )
        # Initial check
        value = getattr(self.property_of, self.property_name)
        if self.predicate(value):
            stream.remove()
            f = Future()
            f.set_result(value)
            return f.__await__()

        return self._stream_future(stream).__await__()


@dataclass(frozen=True, kw_only=True)
class FunctionCondition(BaseCondition[T]):
    function: Callable[..., T]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwds: dict[str, Any] = field(default_factory=dict)

    def __await__(self) -> Generator[T, None, Any]:
        """Defaults to waiting for the function to be truthy."""
        stream = self.conn.add_stream(self.function, *self.args, **self.kwds)
        return self._stream_future(stream).__await__()
