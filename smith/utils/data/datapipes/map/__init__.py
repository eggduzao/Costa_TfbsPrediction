# Functional DataPipe
from smith.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from smith.utils.data.datapipes.map.combinatorics import (
    ShufflerIterDataPipe as Shuffler,
)
from smith.utils.data.datapipes.map.combining import (
    ConcaterMapDataPipe as Concater,
    ZipperMapDataPipe as Zipper,
)
from smith.utils.data.datapipes.map.grouping import BatcherMapDataPipe as Batcher
from smith.utils.data.datapipes.map.utils import (
    SequenceWrapperMapDataPipe as SequenceWrapper,
)


__all__ = ["Batcher", "Concater", "Mapper", "SequenceWrapper", "Shuffler", "Zipper"]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
