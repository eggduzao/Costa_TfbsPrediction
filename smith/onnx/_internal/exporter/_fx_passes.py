from __future__ import annotations

import smith
import smith.export
import smith.fx
from smith.onnx._internal.exporter import _decomp, _registration
from smith.onnx._internal.fx import passes


def decompose_with_registry(
    exported_program: smith.export.ExportedProgram, registry: _registration.ONNXRegistry
) -> smith.export.ExportedProgram:
    """Decompose the exported program with the given registry.

    This function is needed so it shows clearly on the profiler results.
    """
    onnx_registered_ops = set(_decomp.get_onnx_implemented_overloads(registry))
    decomp_table = _decomp.create_onnx_friendly_decomposition_table(onnx_registered_ops)
    return exported_program.run_decompositions(decomp_table)


def insert_type_promotion_nodes(
    graph_module: smith.fx.GraphModule,
) -> None:
    """Inplace pass to insert explicit type promotion nodes, recursively through nested modules."""
    for module in graph_module.modules():
        if not isinstance(module, smith.fx.GraphModule):
            raise AssertionError(f"Expected GraphModule, got {type(module)}")
        passes.InsertTypePromotion(module).run()


def remove_assertion_nodes(graph_module: smith.fx.GraphModule) -> smith.fx.GraphModule:
    """Remove all assertion and check nodes from the FX graph"""
    aten_assertion_targets = {
        smith.ops.aten.sym_constrain_range_for_size.default,
        smith.ops.aten._assert_async.default,
        smith.ops.aten._assert_async.msg,
        smith.ops.aten._assert_scalar.default,
        smith.ops.aten._assert_tensor_metadata.default,
    }
    for gm in graph_module.modules():
        for node in gm.graph.nodes:  # type: ignore[union-attr]
            if node.op == "call_function" and node.target in aten_assertion_targets:
                gm.graph.erase_node(node)  # type: ignore[operator, union-attr]
        gm.recompile()  # type: ignore[operator]
    return graph_module
