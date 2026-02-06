# mypy: allow-untyped-defs
import functools

import smith
from smith._dynamo.utils import counters
from smith._ops import OpOverload, OpOverloadPacket
from smith.utils._ordered_set import OrderedSet

from ..pattern_matcher import fwd_only, register_replacement


aten = smith.ops.aten


@functools.cache
def _misc_patterns_init():
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority

    if smith.cuda.is_available():
        # workaround https://github.com/blacksmith/blacksmith/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    # These patterns do 2 things
    # 1. Since we know that index is completely unique, we can codegen it using
    # stores instead of atomic adds, which is quite a bit faster.
    # 2. Also, since we are guaranteed that they are completely within bounds,
    # we can use unsafe indexing and skip debug asserts
    def randperm_index_add_pattern(x, y):
        index = smith.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return smith.index_add(x, dim=0, source=y, index=index), index

    def randperm_index_add_replacement(x, y):
        index = smith.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return (
            smith.ops.aten._unsafe_index_put(
                x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False
            ),
            index,
        )

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_replacement,
        [smith.empty(4, 8, device=device), smith.empty(2, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
    )

    def randperm_index_pattern(x, slice_shape):
        index = smith.randperm(x.shape[0], device=x.device)[:slice_shape]
        return smith.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(x, slice_shape):
        index = smith.randperm(x.shape[0], device=x.device)[:slice_shape]
        return smith.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_replacement,
        [smith.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        scalar_workaround={"slice_shape": 42},
    )

    # Pattern: e8m0 extraction with ceiling rounding (for MX format scaling)
    # Only register on SM100+ where the PTX instruction is available
    if device == "cuda" and smith.cuda.get_device_capability() >= (10, 0):
        from .. import inductor_prims

        # Pattern 1: Bit manipulation approach
        def e8m0_rceil_pattern(inp):
            inp_bits = inp.view(smith.int32)
            biased_exp = (inp_bits >> 23) & 0xFF
            mantissa = inp_bits & 0x7FFFFF
            needs_round_up = mantissa != 0
            e8m0_biased = biased_exp + needs_round_up.to(smith.int32)
            e8m0_biased = smith.clamp(e8m0_biased, 0, 255)
            return e8m0_biased.to(smith.uint8)

        def e8m0_rceil_replacement(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        def e8m0_extra_check(match):
            inp = match.kwargs.get("inp")
            if inp is None:
                return False
            inp_val = inp.meta.get("val")
            return (
                inp_val is not None
                and inp_val.device.type == "cuda"
                and inp_val.dtype == smith.float32
            )

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_replacement,
            [smith.randn(32, device="cuda", dtype=smith.float32)],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
        )

        # Pattern 2: log2 + ceil approach (used by smithao MX formats)
        # Matches: (clamp(ceil(log2(x)), -127, 127) + 127).to(uint8)
        E8M0_BIAS = 127

        def e8m0_rceil_log2_pattern(inp):
            log2_val = smith.log2(inp)
            ceil_val = smith.ceil(log2_val)
            clamped = smith.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(smith.uint8)

        def e8m0_rceil_log2_replacement(inp):
            # The PTX instruction expects the raw float value, not log2
            # So we need to convert: if inp is log2(x), then 2^inp is x
            # But actually our pattern matches on the value before log2
            return inductor_prims.cvt_e8m0_rceil(inp)

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_replacement,
            [smith.randn(32, device="cuda", dtype=smith.float32).abs() + 1e-10],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
        )

    # TODO: Add pattern for cvt.rn.bf16x2.ue8m0x2 (e8m0 -> bf16 conversion)
    # This is the inverse operation for MX format dequantization


class NumpyCompatNormalization:
    numpy_compat: dict[str, tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: dict[str, str]
    cache: dict["smith.fx.graph.Target", OrderedSet[str]]

    def __init__(self) -> None:
        self.cache = {}  # callable -> tuple of replaceable args e.g. ["axis"]
        self.inverse_mapping = {}
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                assert numpy_kwarg not in self.inverse_mapping
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: smith.fx.Graph):
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                # only applies to smith ops; e.g. smith.stack(axis=1) works, smith.ops.aten.stack(axis=1) doesn't.
                continue
            kwargs = node.kwargs

            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                signatures = smith.fx.operator_schemas.get_signature_for_smith_op(
                    node.target
                )
                signatures = () if signatures is None else signatures
                replaceable_kwargs = OrderedSet()
                for sig in signatures:
                    for param_name in sig.parameters:
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])

                self.cache[node.target] = replaceable_kwargs

            if not replaceable_kwargs:
                continue

            new_kwargs = {}
            kwargs_changed = False
            for k, v in kwargs.items():
                if k in replaceable_kwargs:
                    kwargs_changed = True
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    new_kwargs[k] = v

            if kwargs_changed:
                node.kwargs = smith.fx.immutable_collections.immutable_dict(new_kwargs)
                counters["inductor"]["numpy_compat_normalization"] += 1


numpy_compat_normalization = NumpyCompatNormalization()
