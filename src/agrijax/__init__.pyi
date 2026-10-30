# The names of ``agrijax/__init__.py`` for type checkers and IDEs: the module serves them lazily
# (``__getattr__``), so without this stub an editor cannot complete ``aj.dssat`` or ``aj.calibrate``.
# Keep it in step with ``_LAZY`` there (tests/unit/test_facade.py checks that the two agree).
from typing import Any

from agrijax import _machine, facade_execution
from agrijax import calib as calib
from agrijax import dssat as dssat
from agrijax import facade_export as export
from agrijax.report import plot as plot

__version__: str

calibrate = dssat.calibrate
machine = _machine.machine
options = facade_execution.options
DeviceNotFoundError = facade_execution.DeviceNotFoundError

__all__ = [
    "DeviceNotFoundError",
    "__version__",
    "calib",
    "calibrate",
    "dssat",
    "export",
    "machine",
    "options",
    "plot",
]

def __getattr__(name: str) -> Any: ...
def __dir__() -> list[str]: ...
