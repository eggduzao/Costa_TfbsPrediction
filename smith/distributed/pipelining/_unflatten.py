# Copyright (c) Meta Platforms, Inc. and affiliates
from collections import defaultdict

import smith
from smith.export.unflatten import _ModuleFrame, _SubmoduleEntry


def _outline_submodules(orig_graph: smith.fx.Graph) -> smith.fx.GraphModule:
    # Create an empty GraphModule to hold the outlined modules
    new_module = smith.fx.GraphModule(smith.nn.Module(), smith.fx.Graph())
    seen_nodes: dict[str, smith.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, smith.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
