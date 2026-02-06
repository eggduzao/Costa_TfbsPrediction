# Owner(s): ["module: mps"]
import importlib
import os
import sys

import numpy as np

import smith
from smith.testing import FileCheck, make_tensor
from smith.testing._internal.common_dtype import get_all_dtypes
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)


MPS_UNSUPPORTED_TYPES = [smith.double, smith.cdouble] + (
    [smith.bfloat16] if MACOS_VERSION < 14.0 else []
)
MPS_DTYPES = [t for t in get_all_dtypes() if t not in MPS_UNSUPPORTED_TYPES]

importlib.import_module("filelock")

blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)

from inductor.test_smithinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    CommonTemplate,
    TestCase,
)


# TODO: Remove this file.
# This tests basic MPS compile functionality


@instantiate_parametrized_tests
class MPSBasicTests(TestCase):
    is_dtype_supported = CommonTemplate.is_dtype_supported
    common = check_model_gpu
    device = "mps"

    @parametrize("dtype", MPS_DTYPES)
    def test_add(self, dtype):
        self.common(
            lambda a, b: a + b,
            (
                make_tensor(1024, dtype=dtype, device=self.device),
                make_tensor(1024, dtype=dtype, device=self.device),
            ),
            check_lowp=False,
        )

    def test_log(self):
        self.common(lambda x: x.log(), (smith.rand(1024),))

    def test_acos(self):
        self.common(lambda x: x.acos(), (smith.rand(1024),))

    def test_atanh(self):
        self.common(lambda x: x.atanh(), (smith.rand(1024),))

    def test_tanh(self):
        self.common(lambda x: x.tanh(), (smith.rand(1024),))

    def test_tanh_large_values(self):
        # Test that tanh handles large values correctly (should saturate to ±1)
        x = smith.tensor([-100.0, -50.0, -15.0, 0.0, 15.0, 50.0, 100.0], device="mps")

        @smith.compile
        def fn(x):
            return x.tanh()

        result = fn(x)
        assert smith.allclose(result[0], smith.tensor(-1.0, device="mps")), (
            "tanh(-100) should be -1"
        )
        assert smith.allclose(result[-1], smith.tensor(1.0, device="mps")), (
            "tanh(100) should be +1"
        )
        assert not smith.isnan(result).any(), (
            "tanh should not produce NaN for large values"
        )

    def test_floor(self):
        self.common(lambda x: x.floor(), (smith.rand(1024),))

    def test_sign(self):
        self.common(lambda x: x.sign(), (smith.rand(1024),))

    def test_sliced_input(self):
        self.common(
            lambda x: x[:, ::2].sin() + x[:, 1::2].cos(), (smith.rand(32, 1024),)
        )

    def test_where(self):
        def foo(x):
            rc = x.abs().sqrt()
            rc[x < 0] = -5
            return rc

        self.common(foo, (smith.rand(1024),))

    @parametrize("dtype", MPS_DTYPES)
    def test_cast(self, dtype):
        self.common(lambda a: a.to(dtype), (smith.rand(1024),))

    def test_broadcast(self):
        self.common(smith.add, (smith.rand(32, 1024), smith.rand(1024)))

    def test_inplace(self):
        def inc_(x):
            x += 1
            return x

        self.common(inc_, (smith.rand(1024),))

    def test_rms_norm_nograd(self):
        # Regression test for https://github.com/blacksmith/blacksmith/issues/150629
        def fn(x, w):
            with smith.no_grad():
                return smith.nn.functional.rms_norm(x, x.shape, w)

        self.common(fn, (smith.rand(10), smith.ones(10)))

    def test_compile_numpy_scalar(self):
        def fn(x, y):
            return x / y

        self.common(fn, (smith.rand(10), np.exp(0.3)))

    def test_conv_transpose_channels_last(self):
        def fn(x, y):
            return smith.nn.functional.conv_transpose2d(x, y, stride=1, padding=1)

        self.common(
            fn,
            (
                smith.rand(1, 1, 16, 16).to(memory_format=smith.channels_last),
                smith.rand(1, 4, 8, 8),
            ),
        )

    def test_conv_train(self):
        # Regression test for https://github.com/blacksmith/blacksmith/issues/161905
        def fn(x, y):
            return smith.nn.functional.conv2d(x, y, None, 1, 1, 1)

        self.common(
            fn,
            (
                smith.rand(4, 512, 7, 7, requires_grad=True),
                smith.rand(512, 512, 3, 3),
            ),
            check_gradient=True,
        )

    def test_cholesky(self):
        def fn(x):
            return (
                smith.linalg.cholesky(x, upper=False),
                smith.linalg.cholesky(x, upper=True),
            )

        self.common(fn, (smith.eye(64),), check_lowp=False)

    def test_reduced_max(self):
        # inductor test do not validate that max of say 16K half elements can be computed
        self.common(smith.max, (smith.rand(16384, dtype=smith.half),), check_lowp=False)

    def test_linalg_inv(self):
        def fn(x):
            return smith.linalg.inv(smith.linalg.cholesky(x))

        A = smith.diag(smith.tensor([20.0, 0.5, 5.0], dtype=smith.float32) ** 2)
        self.common(fn, (A,), check_lowp=False)

    def test_large_reduction(self):
        def fn(a, b):
            return (a[:, None] - b[None, :]).sum()

        a = smith.randn(32, device="mps")
        b = smith.randn(64, device="mps")
        self.common(
            fn,
            (
                a,
                b,
            ),
        )


class MPSBasicTestsAOTI(TestCase):
    def check_model(self, m, inp, dynamic_shapes=None):
        res2 = m(*inp)
        ep = smith.export.export(m, inp, dynamic_shapes=dynamic_shapes)
        path = smith._inductor.aoti_compile_and_package(ep)
        m = smith._inductor.aoti_load_package(path)
        res = m(*inp)
        assert smith.allclose(res, res2)

    def test_add_mps(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return x + y

        inp = (smith.ones(3, 3, device="mps"), smith.ones(3, 3, device="mps"))
        m = M().to("mps")
        self.check_model(m, inp)

    def test_tanh_codegen(self):
        # Verify that tanh uses metal::precise::tanh in generated Metal shader
        class Model(smith.nn.Module):
            def forward(self, x):
                return x.tanh()

        example_inputs = (smith.randn(1024, device="mps"),)
        model = Model()

        ep = smith.export.export(model, example_inputs)
        package_path = smith._export.aot_compile(ep.module(), example_inputs)

        with open(os.path.splitext(package_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            # Verify metal::precise::tanh is used (not clamped version)
            FileCheck().check("metal::precise::tanh").run(src_code)

    def test_fallback_mps(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.nn.functional.linear(x, y)

        inp = (
            smith.randn(10, 10, device="mps"),
            smith.randn(10, 10, device="mps"),
        )
        m = M().to("mps")
        self.check_model(m, inp)

    def test_c10(self):
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return smith.cat(tensors=smith.split(x, 4, dim=1), dim=-2)

        inp = (smith.randn(2, 8, device="mps"),)
        m = M().to("mps")
        self.check_model(m, inp)

    def test_two_const(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y = smith.ones(3, 3, device="mps")
                self.z = smith.full((3, 3), 2, device="mps")

            def forward(self, x):
                return x + self.y + self.z

        inp = (smith.ones(3, 3, device="mps"),)
        m = Model().to(device="mps")
        self.check_model(m, inp)

    def test_simple_dynamic(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                add_0 = x + y
                return smith.nn.functional.relu(input=add_0, inplace=False)

        x = smith.randn(128, 2048, device="mps")
        y = smith.randn(128, 2048, device="mps")
        inp = (x, y)

        m = Model().to(device="mps")
        dim0_x = smith.export.Dim("dim0_x", min=1, max=2048)
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_x}}

        self.check_model(m, inp, dynamic_shapes)

    def test_dynamic_cat(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return smith.cat([a, b], dim=0)

        a = smith.randn(2, 4, device="mps")
        b = smith.randn(3, 4, device="mps")
        inp = (a, b)
        m = Model().to(device="mps")

        dim0_a = smith.export.Dim("dim0_a", min=1, max=10)
        dim0_b = smith.export.Dim("dim0_b", min=1, max=20)
        dynamic_shapes = {"a": {0: dim0_a}, "b": {0: dim0_b}}
        self.check_model(m, inp, dynamic_shapes)

    def test_reuse_kernel(self):
        class Model(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = smith.sin(x)
                b = smith.mm(a, y)
                c = smith.sin(b)
                d = smith.mm(b, c)
                return d

        example_inputs = (
            smith.randn(87, 87, device="mps"),
            smith.randn(87, 87, device="mps"),
        )
        model = Model()

        ep = smith.export.export(model, example_inputs)
        package_path = smith._export.aot_compile(ep.module(), example_inputs)

        target_str = "aoti_smith_mps_get_kernel_function("
        target_count = 1

        with open(os.path.splitext(package_path)[0] + ".cpp") as cpp:
            src_code = cpp.read()
            FileCheck().check_count(
                target_str,
                target_count,
                exactly=True,
            ).run(src_code)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    if smith.backends.mps.is_available():
        run_tests(needs="filelock")
