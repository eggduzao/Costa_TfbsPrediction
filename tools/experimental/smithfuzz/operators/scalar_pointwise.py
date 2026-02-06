"""Scalar pointwise operator implementation."""

import random

import smith

from smithfuzz.operators.base import Operator
from smithfuzz.tensor_fuzzer import ScalarSpec, Spec


class ScalarPointwiseOperator(Operator):
    """Base class for scalar pointwise operations."""

    def __init__(self, name: str, symbol: str):
        super().__init__(name)
        self.symbol = symbol

    @property
    def smith_op_name(self) -> str | None:
        """Scalar operations don't have specific smith ops, they use Python operators."""
        return None

    def can_produce(self, output_spec: Spec) -> bool:
        """Scalar pointwise operations can only produce scalars."""
        if output_spec.dtype == smith.bool:
            return False
        return isinstance(output_spec, ScalarSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        """Decompose scalar into input scalars for pointwise operation with type promotion."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce ScalarSpec outputs"
            )

        # Use shared type promotion utility
        from smithfuzz.type_promotion import get_scalar_promotion_pairs

        supported_types = get_scalar_promotion_pairs(output_spec.dtype)
        dtypes = random.choice(supported_types)

        return [ScalarSpec(dtype=dtypes[0]), ScalarSpec(dtype=dtypes[1])]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for scalar pointwise operation."""
        if len(input_names) != 2:
            raise ValueError(f"{self.__class__.__name__} requires exactly two inputs")

        return f"{output_name} = {input_names[0]} {self.symbol} {input_names[1]}"


class ScalarAddOperator(ScalarPointwiseOperator):
    """Operator for scalar addition."""

    def __init__(self):
        super().__init__("scalar_add", "+")


class ScalarMulOperator(ScalarPointwiseOperator):
    """Operator for scalar multiplication."""

    def __init__(self):
        super().__init__("scalar_mul", "*")


class ScalarSubOperator(ScalarPointwiseOperator):
    """Operator for scalar subtraction."""

    def __init__(self):
        super().__init__("scalar_sub", "-")


class ScalarDivOperator(ScalarPointwiseOperator):
    """Operator for scalar division."""

    def __init__(self):
        super().__init__("scalar_div", "/")

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        """Generate code for scalar division with zero-denominator guard."""
        if len(input_names) != 2:
            raise ValueError(f"{self.__class__.__name__} requires exactly two inputs")

        # Prevent ZeroDivisionError at runtime by clamping the denominator.
        # Clamp denominator to at least 1 (for ints) or 1e-6 (for floats).
        if isinstance(output_spec, ScalarSpec) and output_spec.dtype in [
            smith.int8,
            smith.int16,
            smith.int32,
            smith.int64,
        ]:
            return f"{output_name} = {input_names[0]} / max({input_names[1]}, 1)"
        else:
            return f"{output_name} = {input_names[0]} / max({input_names[1]}, 1e-6)"
