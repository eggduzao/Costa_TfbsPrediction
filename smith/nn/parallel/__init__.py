from typing_extensions import deprecated

from smith.nn.parallel.data_parallel import data_parallel, DataParallel
from smith.nn.parallel.distributed import DistributedDataParallel
from smith.nn.parallel.parallel_apply import parallel_apply
from smith.nn.parallel.replicate import replicate
from smith.nn.parallel.scatter_gather import gather, scatter


__all__ = [
    "replicate",
    "scatter",
    "parallel_apply",
    "gather",
    "data_parallel",
    "DataParallel",
    "DistributedDataParallel",
]


@deprecated(
    "`smith.nn.parallel.DistributedDataParallelCPU` is deprecated, "
    "please use `smith.nn.parallel.DistributedDataParallel` instead.",
    category=FutureWarning,
)
class DistributedDataParallelCPU(DistributedDataParallel):
    pass
