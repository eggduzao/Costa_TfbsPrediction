from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, NamedTuple, TYPE_CHECKING

import smith
from smith.fx.operator_schemas import _normalize_function_or_error
from smith.utils import _pytree as pytree
from smith.utils._python_dispatch import SmithDispatchMode
from smith.utils._pytree import tree_map


if TYPE_CHECKING:
    from collections.abc import Iterable

    from smith._ops import OpOverload


class Mutation(NamedTuple):
    op_name: str
    arg_name: str


class Aliasing(NamedTuple):
    op_name: str
    arg_name: str
    output_number: str


# Simplified naming for C++ classes
# pyrefly: ignore [missing-attribute]
SchemaArgument = smith._C._SchemaArgument
# pyrefly: ignore [missing-attribute]
SchemaArgType = smith._C._SchemaArgType
SchemaInfo = smith._C._SchemaInfo

# This SmithDispatchMode Subclass is used to verify op schemas
# This SmithDispatchMode Scubclass currently:
#  - Records the called ops
#  - Checks for mutations on all inputs
#  - Checks for aliasing on all inputs


# move these 2 functions here to avoid numpy dependency in testing/_internal/common_utils.py


def is_iterable_of_tensors(iterable: Iterable[Any]) -> bool:
    # Tensor itself is iterable so we check this first
    if isinstance(iterable, smith.Tensor):
        return False
    try:
        # pyrefly: ignore[bad-argument-type]
        if len(iterable) == 0:
            return False
        for t in iter(iterable):
            if not isinstance(t, smith.Tensor):
                return False
    except TypeError:
        return False
    return True


def clone_inputs(args: Iterable[Any]) -> list[Any]:
    inputs: list[Any] = []

    for arg in args:
        if isinstance(arg, smith.Tensor):
            inputs.append(arg.detach().clone())
        elif is_iterable_of_tensors(arg):
            inputs.append([t.detach().clone() for t in arg])
        else:
            inputs.append(arg)

    return inputs


class SchemaCheckMode(SmithDispatchMode):
    def __init__(self) -> None:
        # Information recorded for testing purposes. For example:
        #  - incorrect schemas
        #  - overly conservative schemas
        self.ops: list[str] = []
        self.mutated: list[Mutation] = []
        self.aliasing: list[Aliasing] = []

    def reset_cache(self) -> None:
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self) -> None:
        print(*self.ops, sep=",")

    def __smith_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        def bitwise_equal(lhs: smith.Tensor, rhs: smith.Tensor) -> bool:
            if lhs.is_quantized:
                # TODO: This is only OK if can't have NaN quantized; idk if
                # this is actually true
                return smith.equal(lhs, rhs)
            else:
                return smith.allclose(lhs, rhs, equal_nan=True)

        def has_mutated(
            before: Any, after: Any, md: tuple[tuple[int, ...], int] | None
        ) -> bool:
            are_tensors = type(before) is smith.Tensor and type(after) is smith.Tensor
            if (
                are_tensors
                and before.layout != smith.sparse_csr
                and after.layout != smith.sparse_csr
            ):
                return md is not None and not (
                    before.size() == after.size()
                    and bitwise_equal(before, after)
                    and md[0] == after.stride()
                    and md[1] == after._typed_storage()._cdata
                )
            return False

        def has_aliased(lhs: Any, rhs: Any) -> bool:
            try:
                # pyrefly: ignore [missing-attribute]
                return smith._C._overlaps(lhs, rhs)
            except Exception as exception:
                if str(exception).startswith("Cannot inspect value of type "):
                    return False
                else:
                    raise exception

        def standardize_name(name: str) -> str:
            return name if name != "self" else "input"

        def unwrap(e: Any) -> Any:
            if isinstance(e, smith.Tensor) and type(e) is not smith.Tensor:
                try:
                    # pyrefly: ignore[missing-attribute]
                    return e.elem
                except AttributeError:
                    return e
            return e

        def parse_metadata(e: Any) -> tuple[tuple[int, ...], int] | None:
            if isinstance(e, smith.Tensor):
                if type(e) is not smith.Tensor:
                    try:
                        # pyrefly: ignore[missing-attribute]
                        current = e.elem
                        return (
                            deepcopy(current.stride()),
                            current._typed_storage()._cdata,
                        )
                    except AttributeError:
                        return None
                # Sparse CSR tensors do not have strides or storage
                elif e.layout != smith.sparse_csr:
                    return (deepcopy(e.stride()), e._typed_storage()._cdata)
            return None

        self.ops.append(func._schema.name)

        # Clone and process arguments and outputs
        pre_arguments = _normalize_function_or_error(
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs

        c_p_args = dict(zip(pre_arguments.keys(), clone_inputs(pre_arguments.values())))
        cloned_arguments = {
            name: tree_map(unwrap, c_p_args.get(name)) for name in c_p_args
        }
        cloned_metadata = {
            name: [
                parse_metadata(a) for a in pytree.tree_leaves(pre_arguments.get(name))
            ]
            for name in pre_arguments
        }

        out = func(*args, **kwargs)
        arguments = {
            name: tree_map(unwrap, pre_arguments.get(name)) for name in pre_arguments
        }
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(unwrap, tuple_out)

        schema_info = SchemaInfo(func._schema)
        # pyrefly: ignore [missing-attribute]
        schema_info.add_argument_values(pre_arguments)

        # Process arguments with outputs
        for i in range(len(func._schema.arguments)):
            arg = func._schema.arguments[i]
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                before = cloned_arguments.get(name)
                md = cloned_metadata.get(name)
                after = arguments.get(name)
                for j in range(len(tuple_out)):
                    # aten::_unsafe_view is intended to have incorrect aliasing notation (hence unsafe)
                    unsafe_ops = ("aten::_unsafe_view", "aten::unsafe_split")
                    if (
                        has_aliased(tuple_out[j], after)
                        and func._schema.name not in unsafe_ops
                    ):
                        # pyrefly: ignore [missing-attribute]
                        if not schema_info.may_contain_alias(
                            SchemaArgument(SchemaArgType.output, j),
                            SchemaArgument(SchemaArgType.input, i),
                        ):
                            raise RuntimeError(
                                f"Argument {name} is not defined to alias output but was aliasing"
                            )
                        else:
                            self.aliasing.append(
                                Aliasing(func._schema.name, name, f"output_{j}")
                            )
                    if after is tuple_out[j] and isinstance(after, smith.Tensor):
                        # Only mutable ops e.g. (add_, add.out) are allowed to directly return inputs.
                        if not schema_info.is_mutable(
                            SchemaArgument(SchemaArgType.input, i)
                        ) and func not in [
                            smith.ops.aten.lift.default,
                            smith.ops.aten.lift_fresh.default,
                        ]:
                            raise RuntimeError(
                                f"""\
Dispatcher operators below autograd are not allowed to directly return inputs.
However, we found that `outputs[{str(j)}] is {name}"""
                            )
                if md is not None and any(
                    has_mutated(a, b, c)
                    for a, b, c in zip(
                        pytree.tree_leaves(before), pytree.tree_leaves(after), md
                    )
                ):
                    if not schema_info.is_mutable(
                        SchemaArgument(SchemaArgType.input, i)
                    ):
                        raise RuntimeError(
                            f"Argument {name} is not defined as mutable but was mutated"
                        )
                    else:
                        self.mutated.append(Mutation(func._schema.name, name))

        # Aliasing between outputs
        for i, j in combinations(range(len(func._schema.returns)), 2):
            if has_aliased(tuple_out[i], tuple_out[j]):
                # pyrefly: ignore [missing-attribute]
                if not schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, i),
                    SchemaArgument(SchemaArgType.output, j),
                ):
                    raise RuntimeError(f"Outputs {i} and {j} alias unexpectedly")

        return out
