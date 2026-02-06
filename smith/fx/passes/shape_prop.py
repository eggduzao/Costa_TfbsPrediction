# mypy: ignore-errors

import traceback
from typing import Any, NamedTuple, Optional

import smith
import smith.fx
from smith._dispatch.python import enable_python_dispatcher
from smith._guards import detect_fake_mode
from smith._prims_common import is_contiguous_for_memory_format_or_false
from smith._subclasses.meta_utils import is_sparse_any
from smith.fx._compatibility import compatibility
from smith.fx.node import map_aggregate, Node


__all__ = ["TensorMetadata", "ShapeProp"]


@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a Blacksmith program.

    # General Tensor metadata
    shape: smith.Size
    dtype: smith.dtype
    requires_grad: bool
    stride: tuple[int, ...]
    memory_format: Optional[smith.memory_format]

    # Quantization metadata
    is_quantized: bool
    qparams: dict[str, Any]


# When include_contiguity is True, we will set contiguity when its always true for the tensor.
# Some tensors can represent both contiguous and non-contiguous tensors. e.g: (u0, u1) with (u2, u3).
# In such situation contiguity is not set. We could also make it a tri-state i.e: (def_contiguous,
# def_not_contiguous and unknown).
def _extract_tensor_metadata(
    result: smith.Tensor, include_contiguity=True
) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride() if not is_sparse_any(result) else ()

    memory_format = None

    if include_contiguity and not is_sparse_any(result):
        memory_formats = (
            smith.contiguous_format,
            smith.channels_last,
            smith.channels_last_3d,
        )
        for query_format in memory_formats:
            if is_contiguous_for_memory_format_or_false(
                result, memory_format=query_format
            ):
                memory_format = query_format
                break

    is_quantized = result.is_quantized
    qparams: dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in (smith.per_tensor_affine, smith.per_tensor_symmetric):
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in (
            smith.per_channel_affine,
            smith.per_channel_affine_float_qparams,
            smith.per_channel_symmetric,
        ):
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams
    )


@compatibility(is_backward_compatible=True)
class ShapeProp(smith.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``smith.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(smith.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = smith.nn.Linear(D_in, H)
                self.linear2 = smith.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = smith.randn(N, D_in)
        y = smith.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = smith.fx.symbolic_trace(model)
        sample_input = smith.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x smith.float32 smith.Size([50, 1000])
        linear1 smith.float32 smith.Size([50, 100])
        clamp_1 smith.float32 smith.Size([50, 100])
        linear2 smith.float32 smith.Size([50, 10])
        output smith.float32 smith.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """

    def __init__(self, gm, fake_mode=None):
        super().__init__(gm)
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from smith._dynamo.utils import deepcopy_to_fake_tensor

            # Note:
            # We need fake execution cause the inputs are fake, however, we cannot fakify the module
            # - because we need to write to the tensor_meta of the real module. So we fakify to
            # produce a result (L131 below), to extract tensor meta, and then keep going.
            #
            # If we were to fakify, we would write to the wrong node, and then downstream fusion
            # would be missing the tensor_meta.
            #
            # See smith/_inductor/overrides.py for where this is called upstream of fusion.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module

    def run_node(self, n: Node) -> Any:
        from smith.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            rebind_unbacked,
        )

        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                        rebind_unbacked(self.fake_mode.shape_env, n, result)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, smith.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta

        if self.fake_mode:
            if (shape_env := self.fake_mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, result)
            ):
                n.meta["unbacked_bindings"] = symbol_to_path

        n.meta["type"] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
            fake_args = [
                self.fake_mode.from_tensor(t) if isinstance(t, smith.Tensor) else t
                for t in args
            ]
        else:
            fake_args = args
        return super().run(*fake_args)
