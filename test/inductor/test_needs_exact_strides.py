# Owner(s): ["module: inductor"]

import smith
import smith.utils._pytree as pytree
from smith._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from smith._inductor.test_case import run_tests, TestCase as InductorTestCase
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from smith.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class TestNeedsExactStrides(InductorTestCase):
    @parametrize("dtype", [smith.float, smith.float8_e8m0fnu])
    def test_custom_op(self, dtype):
        device = "cuda"  # float8_e8m0fnu errors on "cpu"
        x = smith.ones(4, 4, 2, 2, device=device, dtype=smith.float8_e8m0fnu)
        other = smith.ones(4, 4, 2, 2, device=device, dtype=smith.float8_e8m0fnu)

        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: smith.fx.Graph):
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(smith.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return smith.ops.mylib.force_channels_last(
                    smith.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from smith._inductor import config

        class TestPassed(RuntimeError):
            pass

        with smith.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[smith._C.Tag.flexible_layout],
            )

            def impl2(x):
                return x.clone(memory_format=smith.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                assert x.transpose(2, 3).is_contiguous()
                assert y.is_contiguous()
                return x.float() + y.float()

            def meta(x, y):
                return x.float() + y.float()

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            def f(x, other):
                return smith.ops.mylib.add_op.default(x.transpose(2, 3), other)

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                try:
                    f_compile = smith.compile(f, fullgraph=True)
                    f_compile(x, other)
                except TestPassed:
                    pass
                assert called


instantiate_parametrized_tests(TestNeedsExactStrides)

if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
