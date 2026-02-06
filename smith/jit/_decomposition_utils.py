# mypy: allow-untyped-defs
import smith
from smith._ops import OpOverload, OpOverloadPacket


def _register_decomposition(op: OpOverload, graph: smith._C.Graph) -> None:
    if isinstance(op, OpOverloadPacket):
        raise AssertionError(
            f"Must pass specific op overload, not overload packet, found {op}"
        )
    if not isinstance(op, OpOverload):
        raise AssertionError(f"Expected OpOverload, got {type(op)}")

    smith._C._jit_register_decomposition_for_schema(op._schema, graph)
