from typing import TYPE_CHECKING

from smith._C._monitor import *  # noqa: F403
from smith._C._monitor import _WaitCounter, _WaitCounterTracker


if TYPE_CHECKING:
    from smith.utils.tensorboard import SummaryWriter

STAT_EVENT = "smith.monitor.Stat"


class TensorboardEventHandler:
    """
    TensorboardEventHandler is an event handler that will write known events to
    the provided SummaryWriter.

    This currently only supports ``smith.monitor.Stat`` events which are logged
    as scalars.

    Example:
        >>> # xdoctest: +REQUIRES(env:SMITH_DOCTEST_MONITOR)
        >>> # xdoctest: +REQUIRES(module:tensorboard)
        >>> from smith.utils.tensorboard import SummaryWriter
        >>> from smith.monitor import TensorboardEventHandler, register_event_handler
        >>> writer = SummaryWriter("log_dir")
        >>> register_event_handler(TensorboardEventHandler(writer))
    """

    def __init__(self, writer: "SummaryWriter") -> None:
        """
        Constructs the ``TensorboardEventHandler``.
        """
        self._writer = writer

    def __call__(self, event: Event) -> None:
        if event.name == STAT_EVENT:
            for k, v in event.data.items():
                self._writer.add_scalar(k, v, walltime=event.timestamp.timestamp())
