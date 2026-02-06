# Owner(s): ["module: inductor"]
import contextlib
import operator
from collections import defaultdict

import smith
import smith._inductor.pattern_matcher as pattern_matcher
import smith.fx as fx
from smith._dynamo.utils import counters
from smith._inductor import config
from smith._inductor.codegen.common import get_custom_backend_pass_for_device
from smith._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    get_hash_for_files,
)
from smith._inductor.lowering import lowerings as L
from smith._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_utils import IS_LINUX
from smith.testing._internal.inductor_utils import HAS_CPU, patch_inductor_backend


@config.patch({"freezing": True})
class TestCustomPassBase(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, smith.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count,
        matcher_nodes,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        counters.clear()
        maybe_autocast = contextlib.nullcontext()
        with smith.no_grad(), maybe_autocast:
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual = smith.compile(mod)(*clone_inputs)
            smith.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            self.assertEqual(
                counters["inductor"]["pattern_matcher_count"], matcher_count
            )
            self.assertEqual(
                counters["inductor"]["pattern_matcher_nodes"],
                matcher_nodes,
            )


aten = smith.ops.aten
mkldnn = smith.ops.mkldnn


def change_cos_pass(graph):
    for node in graph.nodes:
        if node.op == "call_function" and node.target == aten.cos.default:
            node.target = aten.sin.default


class ChangeCosCustomPass(CustomGraphPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: smith.fx.graph.Graph):
        change_cos_pass(g)

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,))


class TestPostGradCustomPrePostPass(TestCustomPassBase):
    #  mkldnn fusion's pattern_matcher
    # (smith/_inductor/fx_passes/mkldnn_fusion.py),
    # and apply it to custom post_grad_passes.
    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        # pattern
        def _mkldnn_conv_relu_pattern():
            return CallFunction(
                aten.relu,
                CallFunction(
                    mkldnn._convolution_pointwise.default,
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    _users=1,
                ),
            )

        # utils of pattern matcher registration
        def _register_fusion_lowering(pattern, custom_pass_dict):
            def dummy_check(m):
                return True

            def register_custom_lowering_pattern(
                pattern, extra_check, custom_pass_dict
            ):
                return pattern_matcher.register_lowering_pattern(
                    pattern, extra_check, pass_dict=custom_pass_dict
                )

            @register_custom_lowering_pattern(pattern, dummy_check, custom_pass_dict)
            def fn(match, *args, **kwargs):
                computation_args = list(args)[:-3] + ["relu", [], ""]
                return L[mkldnn._convolution_pointwise.default](*computation_args)

            return fn

        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    # custom post grad pass
    class _CustomPass(PatternMatcherPass, CustomGraphPass):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, g: smith.fx.graph.Graph):
            self.apply(g)

        def uuid(self) -> bytes:
            return get_hash_for_files((__file__,))

    # case model
    class _ConvReLU(smith.nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.conv = smith.nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x1 = self.conv(x)
            return x1.relu()

    def test_custom_joint_pass_pre(self):
        with config.patch(joint_custom_pre_pass=ChangeCosCustomPass()):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = smith.randn(8, dtype=smith.float32)
            smith.testing.assert_close(smith.compile(f)(x), g(x))

    def test_custom_joint_pass_post(self):
        with config.patch(joint_custom_post_pass=ChangeCosCustomPass()):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = smith.randn(8, dtype=smith.float32)
            smith.testing.assert_close(smith.compile(f)(x), g(x))

    def test_custom_pre_pass(self):
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            post_grad_custom_pre_pass=self._CustomPass(),
            # define pattern match as custom post grad opt pass
            post_grad_custom_post_pass=None,
        ):
            # init mkldnn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = smith.randn((1, 16, 56, 56), dtype=smith.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_post_pass(self):
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            # define pattern match as custom post grad opt pass
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=self._CustomPass(),
        ):
            # init mkldnn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = smith.randn((1, 16, 56, 56), dtype=smith.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_pre_grad_pass(self):
        saved_graph = [None]

        def merge_mm_shared_rhs(graph: fx.Graph):
            """
            Bad POC of merging mm with a shared RHS.
            i.e. [mm(x, W), mm(x2, W)] => mm(cat(x, x2), W).split()

            Isn't actually safe for a couple reasons. For example, it doesn't handle the
            case where the LHS inputs depend on each other
            """
            saved_graph[0] = graph
            matmuls = [n for n in graph.nodes if n.target == smith.mm]
            rhs_vals = defaultdict(set)
            for m in matmuls:
                rhs_vals[m.args[1]].add(m)

            order = {n: idx for idx, n in enumerate(graph.nodes)}

            for rhs, matmuls in rhs_vals.items():
                if len(matmuls) == 1:
                    continue
                matmuls = sorted(matmuls, key=lambda x: order[x])
                with graph.inserting_before(matmuls[0]):
                    lhs_vals = [m.args[0] for m in matmuls]
                    new_cat = graph.create_node(
                        "call_function", smith.cat, args=(lhs_vals, 0)
                    )
                    new_mm = graph.create_node(
                        "call_function", smith.mm, args=(new_cat, rhs)
                    )
                    split_vals = graph.create_node(
                        "call_function",
                        smith.split,
                        args=(
                            new_mm,
                            [l.meta["example_value"].shape[0] for l in lhs_vals],
                        ),
                    )
                for idx, m in enumerate(matmuls):
                    m.target = operator.getitem
                    m.args = (split_vals, idx)

        @config.patch(pre_grad_custom_pass=merge_mm_shared_rhs)
        def inner_test():
            @smith.compile
            def f(W, nested_seqs):
                outs = [smith.mm(s, W) for s in nested_seqs]
                return outs

            W = smith.randn(16, 16, dtype=smith.bfloat16)
            nested_seqs = [
                smith.randn(l, 16, dtype=smith.bfloat16) for l in [4, 8, 5, 3]
            ]

            f(W, nested_seqs)
            assert saved_graph[0] is not None
            matmuls = [n for n in saved_graph[0].nodes if n.target == smith.mm]
            assert len(matmuls) == 1

        inner_test()

    def test_custom_backend_pass(self):
        class CustomBackendPass(CustomGraphModulePass):
            def __init__(self, existing_pass: CustomGraphModulePass = None):
                super().__init__()
                self.existing_pass = existing_pass

            def __call__(self, gm: fx.GraphModule) -> None:
                if self.existing_pass:
                    self.existing_pass(gm)

                change_cos_pass(gm.graph)

            def uuid(self) -> bytes:
                return get_hash_for_files((__file__,))

        custom_backend_pass = CustomBackendPass(
            get_custom_backend_pass_for_device("cpu")
        )
        with patch_inductor_backend("cpu", custom_pass=custom_backend_pass):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = smith.randn(8, dtype=smith.float32)
            smith.testing.assert_close(smith.compile(f)(x), g(x))


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and smith.backends.mkldnn.is_available():
        run_tests()
