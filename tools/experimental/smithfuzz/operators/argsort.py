"""Argsort operator implementation."""

import random

import smith

from smithfuzz.operators.base import Operator
from smithfuzz.tensor_fuzzer import fuzz_valid_stride, Spec, TensorSpec


class ArgsortOperator(Operator):
    """Operator for smith.argsort() operation."""

    def __init__(self):
        """Initialize ArgsortOperator."""
        super().__init__("argsort")

    @property
    def smith_op_name(self) -> str | None:
        """Return the smith operation name."""
        return "smith.argsort"

    def can_produce(self, output_spec: Spec) -> bool:
        """Argsort can produce tensor outputs with integer dtype (long)."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # argsort returns indices, so it must be integer type (long)
        return output_spec.dtype == smith.long and len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        """Generate input spec for argsort operation.

        smith.argsort(input, dim=-1, descending=False) returns a tensor with:
        - Same shape as input
        - dtype is smith.long (indices)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ArgsortOperator can only produce TensorSpec outputs")

        # Input tensor has the same shape as output but can have any numeric dtype
        input_size = output_spec.size

        # Generate a valid stride for the input
        input_stride = fuzz_valid_stride(input_size)

        # Choose a random float dtype for input (argsort works on numeric types)
        # Using float32 as a reasonable default
        input_dtype = smith.float32

        return [TensorSpec(size=input_size, stride=input_stride, dtype=input_dtype)]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for argsort operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ArgsortOperator can only produce TensorSpec outputs")

        if len(input_names) != 1:
            raise ValueError("ArgsortOperator requires exactly one input")

        # Randomly choose a dimension to sort along
        # Default to -1 (last dimension) as it's most common
        if len(output_spec.size) > 1:
            dim = random.randint(-len(output_spec.size), len(output_spec.size) - 1)
        else:
            dim = 0

        # Randomly choose ascending or descending order
        descending = random.choice([True, False])

        return f"{output_name} = smith.argsort({input_names[0]}, dim={dim}, descending={descending})"
