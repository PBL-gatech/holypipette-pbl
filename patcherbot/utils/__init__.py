from .RecordingStateManager import RecordingStateManager
from .config import Config, NumberWithUnit, Number, Boolean, Selector
from .log_utils import LoggingObject, setup_logging
from .exception_handler import set_global_exception_hook


def __getattr__(name):
    if name == "FileLogger":
        from .FileLogger import FileLogger

        return FileLogger
    if name == "EPhysLogger":
        from .EPhysLogger import EPhysLogger

        return EPhysLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Boolean",
    "Config",
    "EPhysLogger",
    "FileLogger",
    "LoggingObject",
    "Number",
    "NumberWithUnit",
    "RecordingStateManager",
    "Selector",
    "set_global_exception_hook",
    "setup_logging",
]

