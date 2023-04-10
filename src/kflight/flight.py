import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, overload
from functools import partial

from krpc.services.spacecenter import Vessel, Control, Orbit, Part
from krpc import Client, services, connect

from kflight.conditions import PropertyCondition, FunctionCondition

__all__ = ("Flight",)

from kflight.event_handler import EventHandler, Priority

from kflight.throttle_control import ThrottleControl

T = TypeVar("T")

_log = logging.getLogger(__name__)


class Flight(EventHandler, ThrottleControl):
    """The main class for the flight program."""

    def __init__(self):
        super().__init__()
        self.connection: Client | None = None
        self._detach: bool = False
        self._plans: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

    @property
    def mechjeb(self) -> services.MechJeb:
        """The MechJeb service."""
        return self.connection.mechjeb

    @property
    def vessel(self) -> Vessel:
        """The active vessel."""
        return self.connection.space_center.active_vessel

    @property
    def control(self) -> Control:
        """The vessel's control."""
        return self.vessel.control

    @property
    def ui(self) -> services.UI:
        """The UI service."""
        return self.connection.ui

    @property
    def sc(self) -> services.SpaceCenter:
        """The SpaceCenter service."""
        return self.connection.space_center

    @property
    def orbit(self) -> Orbit:
        """The vessel's orbit."""
        return self.vessel.orbit

    @property
    def current_game_scene(self) -> str:
        """The current game scene."""
        return self.connection.krpc.current_game_scene

    def active_engines(self) -> list[Part]:
        """The active engines."""
        return [p.part for p in self.vessel.parts.engines if p.active]

    def altitude_agl(self) -> float:
        """
        The altitude above ground. Measured from the root part and NOT the center of mass.
        This should match the altitude shown on the in-game ticker.
        """
        root_part = self.vessel.parts.root
        altitude = self.vessel.flight(root_part.reference_frame).surface_altitude
        return altitude

    def _register_callbacks(self):
        # on_scene_change
        # s1 = self.connection.add_stream(getattr, self.connection.krpc, "current_game_scene")
        # s1.add_callback(lambda v: self.loop.create_task(self.on_scene_change(v)))

        # on_abort
        s2 = self.connection.add_stream(getattr, self.vessel.control, "abort")

        def on_abort(v: bool):
            if v:
                print("on abort called!")
                self._cancel_below(self.loop, Priority.ABORT)
                self._schedule_event(self.on_abort, priority=Priority.ABORT)

        s2.add_callback(on_abort)
        s2.start(wait=False)

    def event(
        self, coro: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Registers an event."""
        if not asyncio.iscoroutinefunction(coro):
            raise TypeError(f"{coro.__name__!r} is not a coroutine function.")
        setattr(self, coro.__name__, coro)
        return coro

    async def on_scene_change(self):
        """Called when the game scene is entered."""
        pass

    @overload
    def wait_until(self, remote_property: T) -> PropertyCondition[T]:
        """Waits until the given condition is true."""

    @overload
    def wait_until(
        self, remote_func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> FunctionCondition[T]:
        """Waits until the given condition is true."""

    def wait_until(self, remote_property, *args, **kwargs):
        """Waits until the given condition is true."""
        # Use function form if callable
        if callable(remote_property):
            return FunctionCondition(
                conn=self.connection,
                function=remote_property,
                args=args,
                kwds=kwargs,
            )
        return PropertyCondition.from_statement(
            self.connection, remote_property, frame=2
        )

    def plan(
        self, coro: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Registers a flight plan."""
        if not asyncio.iscoroutinefunction(coro):
            raise TypeError(f"{coro.__name__!r} is not a coroutine function.")
        name = coro.__name__
        if name in self._plans:
            raise ValueError(f"Plan {name!r} already exists.")
        self._plans[name] = coro
        _log.debug("Registered flight plan %r", name)
        return coro

    def start_plan(
        self, name: str | Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs
    ):
        """Starts a flight plan."""
        if asyncio.iscoroutinefunction(name):
            name = name.__name__
        coro = self._plans[name]
        return self._schedule_event(
            coro, args=args, kwds=kwargs, priority=Priority.NORMAL
        )

    def _ensure_loop(self):
        """Ensures that the event loop is running."""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()

    def close(self) -> None:
        """Closes the connection and event loop."""
        # Cancel events
        self._cancel_below(self.loop, Priority.SYSTEM_ERROR)
        # Close event loop
        self.loop.stop()
        self.loop.close()
        # Close connection
        self.connection.close()

    def run(
        self,
        name: str | None = None,
        host: str = "localhost",
        rpc_port: int = 50000,
        stream_port: int = 50001,
        main: str | Callable[..., Coroutine[Any, Any, Any]] = "main",
        detach: bool = False,
    ) -> None:
        """
        Starts the event loop. This will block until the program is terminated.

        Args:
            name: The name of the program. Defaults to random.
            host: The host to connect to. Defaults to localhost.
            rpc_port: The RPC port to connect to.
            stream_port: The stream port to connect to.
            main: The entry-point coroutine function.
            detach: If True, keep connection open after function returns.
        """
        self.connection = connect(
            name=name, address=host, rpc_port=rpc_port, stream_port=stream_port
        )
        self.loop = asyncio.get_event_loop()
        self._detach = detach

        self._register_callbacks()

        # Skip entry if detach
        if detach:
            return

        # Entrypoint
        if asyncio.iscoroutinefunction(main):
            self._schedule_event(main, priority=Priority.NORMAL)
        elif main in self._plans:
            self.start_plan(main)
        elif (coro_func := getattr(self, main, None)) is not None:
            self._schedule_event(coro_func, priority=Priority.NORMAL)
        else:
            raise ValueError(f"Unknown entrypoint {main!r}.")

        self.loop.run_forever()
