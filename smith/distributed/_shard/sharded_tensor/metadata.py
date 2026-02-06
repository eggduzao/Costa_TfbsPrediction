# mypy: allow-untyped-defs
from dataclasses import dataclass, field
from enum import Enum

import smith
from smith.distributed._shard.metadata import ShardMetadata


class MEM_FORMAT_ENCODING(Enum):
    SMITH_CONTIGUOUS_FORMAT = 0
    SMITH_CHANNELS_LAST = 1
    SMITH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: smith.dtype = field(default=smith.get_default_dtype())
    layout: smith.layout = field(default=smith.strided)
    requires_grad: bool = False
    memory_format: smith.memory_format = field(default=smith.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        # Since smith.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == smith.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.SMITH_CONTIGUOUS_FORMAT
        elif memory_format == smith.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.SMITH_CHANNELS_LAST
        elif memory_format == smith.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.SMITH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid smith.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.SMITH_CONTIGUOUS_FORMAT:
            memory_format = smith.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.SMITH_CHANNELS_LAST:
            memory_format = smith.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.SMITH_PRESERVE_FORMAT:
            memory_format = smith.preserve_format
        else:
            raise RuntimeError(
                f"Invalid smith.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: smith.Tensor) -> "TensorProperties":
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=smith.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )


@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: list[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: smith.Size = field(default=smith.Size([]))

    tensor_properties: TensorProperties = field(default_factory=TensorProperties)
