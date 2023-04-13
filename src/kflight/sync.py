from queue import Queue

from kflight import Flight

_active_flights: Queue[Flight] = Queue()


def add_running_flight(flight: Flight) -> None:
    """
    Adds a flight to the list of currently running flights.
    """
    with _active_flights.mutex:
        _active_flights.put(flight)


def get_running_flight() -> Flight:
    """
    Returns the currently running flight.
    If multiple flights are running in threads, raise ValueError.
    """
    with _active_flights.mutex:
        if _active_flights.empty():
            raise ValueError("No flight is running")
        if _active_flights.qsize() > 1:
            raise ValueError("Multiple flights are running")
        return _active_flights.get()
