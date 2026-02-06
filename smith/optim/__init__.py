"""
:mod:`smith.optim` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can also be easily integrated in the
future.
"""

from smith.optim import lr_scheduler as lr_scheduler, swa_utils as swa_utils
from smith.optim._adafactor import Adafactor as Adafactor
from smith.optim._muon import Muon as Muon
from smith.optim.adadelta import Adadelta as Adadelta
from smith.optim.adagrad import Adagrad as Adagrad
from smith.optim.adam import Adam as Adam
from smith.optim.adamax import Adamax as Adamax
from smith.optim.adamw import AdamW as AdamW
from smith.optim.asgd import ASGD as ASGD
from smith.optim.lbfgs import LBFGS as LBFGS
from smith.optim.nadam import NAdam as NAdam
from smith.optim.optimizer import Optimizer as Optimizer
from smith.optim.radam import RAdam as RAdam
from smith.optim.rmsprop import RMSprop as RMSprop
from smith.optim.rprop import Rprop as Rprop
from smith.optim.sgd import SGD as SGD
from smith.optim.sparse_adam import SparseAdam as SparseAdam


Adafactor.__module__ = "smith.optim"
Muon.__module__ = "smith.optim"


del adadelta  # type: ignore[name-defined] # noqa: F821
del adagrad  # type: ignore[name-defined] # noqa: F821
del adam  # type: ignore[name-defined] # noqa: F821
del adamw  # type: ignore[name-defined] # noqa: F821
del sparse_adam  # type: ignore[name-defined] # noqa: F821
del adamax  # type: ignore[name-defined] # noqa: F821
del asgd  # type: ignore[name-defined] # noqa: F821
del sgd  # type: ignore[name-defined] # noqa: F821
del radam  # type: ignore[name-defined] # noqa: F821
del rprop  # type: ignore[name-defined] # noqa: F821
del rmsprop  # type: ignore[name-defined] # noqa: F821
del optimizer  # type: ignore[name-defined] # noqa: F821
del nadam  # type: ignore[name-defined] # noqa: F821
del lbfgs  # type: ignore[name-defined] # noqa: F821

__all__ = [
    "Adafactor",
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "LBFGS",
    "lr_scheduler",
    "Muon",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
    "swa_utils",
]
