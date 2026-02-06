# Owner(s): ["module: inductor"]
# This test requires libaoti_custom_ops.so to be built, which happens when BUILD_TEST = 1
import logging
import os
import sys
import unittest

import smith
import smith._export
import smith._inductor
import smith._inductor.config
from smith._inductor import config
from smith._inductor.test_case import TestCase
from smith.export import Dim, export
from smith.testing._internal import common_utils
from smith.testing._internal.common_utils import (
    find_library_location,
    IS_CI,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    skipIfXpu,
)
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
from smith.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from smith.utils._python_dispatch import SmithDispatchMode


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_smithinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/funcsmith/filelock")

try:
    try:
        from .test_aot_inductor_utils import (
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from .test_smithinductor import copy_tests, TestFailure
    except ImportError:
        from test_aot_inductor_utils import (  # @manual=fbcode//caffe2/test/inductor:aot_inductor_utils-library
            check_model,
            check_model_with_multiple_inputs,
            code_check_count,
        )
        from test_smithinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            copy_tests,
            TestFailure,
        )
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)
    raise


@smith.library.custom_op(
    "aoti_custom_ops::fn_with_incorrect_optional_tensor", mutates_args=()
)
def fn_with_incorrect_optional_tensor(
    x: smith.Tensor, y: smith.Tensor, z: smith.Tensor
) -> smith.Tensor:
    if z is None:
        return x + y
    else:
        return x + y + z


@fn_with_incorrect_optional_tensor.register_fake
def fn_with_incorrect_optional_tensor_fake(
    x: smith.Tensor, y: smith.Tensor, z: smith.Tensor
) -> smith.Tensor:
    if z is None:
        return x + y
    else:
        return x + y + z


@smith.library.custom_op(
    "aoti_custom_ops::fn_ret_list_of_single_tensor", mutates_args={}
)
def fn_ret_list_of_single_tensor(x: smith.Tensor) -> list[smith.Tensor]:
    s = x.sum().to(smith.int64)
    return [smith.randn(s.item())]


@fn_ret_list_of_single_tensor.register_fake
def _(x):
    ctx = smith._custom_op.impl.get_ctx()
    i0 = ctx.new_dynamic_size()
    return [smith.randn(i0)]


@smith.library.custom_op("aoti_custom_ops::fn_ret_single_tensor", mutates_args={})
def fn_ret_single_tensor(x: smith.Tensor) -> smith.Tensor:
    s = x.sum().to(smith.int64)
    return smith.randn(s.item())


@fn_ret_single_tensor.register_fake
def _(x):
    ctx = smith._custom_op.impl.get_ctx()
    i0 = ctx.new_dynamic_size()
    return smith.randn(i0)


class AOTInductorTestsTemplate:
    def test_custom_op_add(self) -> None:
        class M(smith.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                self.w = smith.randn(3, 3, device=device)

            def forward(self, x):
                const = smith.tensor([1], device=self.device)
                x = smith.ops.aoti_custom_ops.custom_add(x, const)
                return smith.ops.aoti_custom_ops.custom_add(x, self.w)

        m = M(self.device).to(device=self.device)
        args = (smith.randn(3, 3, device=self.device),)
        self.check_model(m, args)

    def test_custom_op_add_output_path(self) -> None:
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.ops.aoti_custom_ops.custom_add(x, y)

        m = M().to(device=self.device)
        args = (
            smith.randn(3, 3, device=self.device),
            smith.randn(3, 3, device=self.device),
        )
        with config.patch("aot_inductor.output_path", "model.pt2"):
            with self.assertRaises(Exception):
                self.check_model(m, args)

    def test_fn_with_optional_tensor_output(self) -> None:
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.ops.aoti_custom_ops.fn_with_optional_tensor_output(x, y)

        m = M().to(device=self.device)
        args = (
            smith.randn(3, 3, device=self.device),
            smith.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_fn_with_optional_tensor_output_2(self) -> None:
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.ops.aoti_custom_ops.fn_with_optional_tensor_output_2(x, y)

        m = M().to(device=self.device)
        args = (
            smith.randn(3, 3, device=self.device),
            smith.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_fn_with_optional_tensor_nullopt_output(self) -> None:
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.ops.aoti_custom_ops.fn_with_optional_tensor_nullopt_output(
                    x, y
                )

        m = M().to(device=self.device)
        args = (
            smith.randn(3, 3, device=self.device),
            smith.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_fn_with_int_output(self) -> None:
        class M(smith.nn.Module):
            def forward(self, x, y):
                i = x.shape[0]
                z, _, _, i1, i2 = smith.ops.aoti_custom_ops.fn_with_int_output(x, y, i)
                return z, z * (i1 + i2 + i)

        m = M().to(device=self.device)
        args = (
            smith.randn(3, 3, device=self.device),
            smith.randn(3, 3, device=self.device),
        )
        self.check_model(m, args)

    def test_custom_op_all_inputs(self) -> None:
        class MyModel(smith.nn.Module):
            # pyre-fixme[3]: Return type must be annotated.
            def __init__(self):
                super().__init__()

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def forward(self, x, y):
                with smith.no_grad():
                    x_dim0 = x.shape[0]
                    x_dim1 = x.shape[1]
                    y_dim0 = y.shape[0]
                    y_dim1 = y.shape[1]
                    symint_0 = x_dim0 + x_dim1
                    symint_1 = y_dim0 * y_dim1

                    z = smith.concat((x, x))

                    _2547 = smith.ops.aoti_custom_ops.fn_with_all_inputs(
                        tensor=x,
                        tensors=[x, y],
                        optional_tensors=[None, z],
                        b8=False,
                        b8s=[True, False],
                        i64=42,
                        i64s=[16, 17],
                        symint=symint_0,
                        symints=[symint_0, symint_1],
                        f64=3.14,
                        f64s=[2.2, 3.3],
                        scalar=1.23,
                        scalars=[45, 67],
                        string="hello",
                        strings=["ab", "cde"],
                        # dtype=smith.float16,
                        # memory_format=smith.contiguous_format,
                        # layout=smith.strided,
                        device=smith.device("cpu"),
                        # optional
                        o_tensor=None,
                        o_tensors=[x, y],
                        o_b8=False,
                        o_b8s=[True, False],
                        o_i64=None,
                        o_i64s=[16, 17],
                        o_symint=symint_1,
                        o_symints=[symint_1, symint_0],
                        o_f64=3.14,
                        o_f64s=None,
                        o_scalar=None,
                        o_scalars=[89, 910],
                        o_string="hello",
                        o_strings=["ab", "cde"],
                        # o_dtype=None,
                        # o_memory_format=smith.contiguous_format,
                        # o_layout=smith.strided,
                        o_device=None,
                    )

                return _2547

        m = MyModel().to(device=self.device)
        x = smith.zeros(4, 8, device=self.device)
        y = smith.ones(3, 9, device=self.device)
        args = (x, y)
        m(*args)

        self.check_model(m, args)

    def test_custom_op_with_multiple_outputs(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x, y):
                out = x + y
                # tuple of Tensor output
                out3, out4 = smith.ops.aoti_custom_ops.fn_with_tuple_output(out, 1)
                # TensorList output
                out5, out6 = smith.ops.aoti_custom_ops.fn_with_list_output(
                    [out3, out4], 1
                )
                # tuple of Tensor and TensorList
                out7, [out8, out9] = smith.ops.aoti_custom_ops.fn_with_mix_outputs(
                    out5, [out6, out4]
                )
                return out3, out4, out5, out6, out7, out8, out9

        m = Model().to(device=self.device)
        args = (
            smith.randn(4, 4, device=self.device),
            smith.randn(4, 4, device=self.device),
        )
        m(*args)

        self.check_model(m, args)

    def test_custom_op_out_variant_without_return(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x, y):
                smith.ops.aoti_custom_ops.fn_out_variant_without_return(x, y)
                return y

        m = Model().to(device=self.device)
        args = (
            smith.randn(10, 10, device=self.device),
            smith.randn(10, 10, device=self.device),
        )
        m(*args)

        self.check_model(m, args)

    def test_custom_op_with_reinterpret_view_inputs(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x):
                out = x.permute([1, 0])
                return smith.ops.aoti_custom_ops.fn_with_default_input(out, 1)

        m = Model().to(device=self.device)
        args = (smith.randn(2, 3, device=self.device),)

        self.check_model(m, args)

    def test_custom_op_with_concat_inputs(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x, y):
                out = smith.concat([x, y], dim=0)
                return smith.ops.aoti_custom_ops.fn_with_default_input(out, 1)

        m = Model().to(device=self.device)
        args = (
            smith.randn(2, 3, device=self.device),
            smith.randn(2, 3, device=self.device),
        )

        self.check_model(m, args)

    def test_custom_op_missing_arg_with_default_value(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x):
                # missing second arg
                return smith.ops.aoti_custom_ops.fn_with_default_input(x)

        m = Model().to(device=self.device)
        args = (smith.randn(2, 3, device=self.device),)

        self.check_model(m, args)

    def test_custom_op_return_list_of_single_tensor(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.ops.aoti_custom_ops.fn_ret_list_of_single_tensor(x)[0] + 1

        m = Model().to(device=self.device)
        args = (smith.randn(3, 4),)
        self.check_model(m, args)

    def test_custom_op_return_single_tensor(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.ops.aoti_custom_ops.fn_ret_single_tensor(x) + 1

        m = Model().to(device=self.device)
        args = (smith.randn(3, 4),)
        self.check_model(m, args)

    @unittest.skipIf(IS_FBCODE, "FbProxyExecutor doesn't have these error msgs")
    def test_incorrect_custom_op_schema(self):
        class M(smith.nn.Module):
            def forward(self, x, y):
                return smith.ops.aoti_custom_ops.fn_with_incorrect_optional_tensor(
                    x, y, None
                )

        m = M().to(device=self.device)
        args = (
            smith.randn(2, 3, device=self.device),
            smith.randn(2, 3, device=self.device),
        )

        with self.assertRaisesRegex(RuntimeError, "Expected extern kernel"):
            self.check_model(m, args)

    def test_boxed_run_inputs_clearing(self):
        # Borrowed from test_smithinductor
        class Model(smith.nn.Module):
            def forward(self, x, y):
                return smith.ops.aoti_custom_ops.custom_add(x, y)

        inps = [
            smith.rand(5, 5, device=self.device),
            smith.rand(5, 5, device=self.device),
        ]
        model = Model().to(device=self.device)
        # NOTE: There are additional references to inps if we use
        # strict=True here, which will cause inps not deallocated
        # in time later in this test.
        ep = smith.export.export(model, tuple(inps), strict=False)
        package = smith._inductor.aoti_compile_and_package(ep)
        fn_compiled = smith._inductor.aoti_load_package(package)

        test_self = self
        sentinel_seen = False

        class TestRefMode(SmithDispatchMode):
            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs if kwargs else {}
                nonlocal inps
                nonlocal test_self
                nonlocal sentinel_seen
                if func is smith.ops.aoti_custom_ops.custom_add.default:
                    # inputs should be deallocated by this point
                    sentinel_seen = True
                    test_self.assertEqual(len(inps), 0)

                return func(*args, **kwargs)

        with TestRefMode():
            fn_compiled.loader.boxed_run(inps)

        self.assertEqual(len(inps), 0)
        self.assertTrue(sentinel_seen)

    @skipIfXpu
    @unittest.skipIf(IS_FBCODE, "unable to find library -laoti_custom_ops")
    def test_custom_op_square(self) -> None:
        class Model(smith.nn.Module):
            def forward(self, x):
                return smith.ops.aoti_custom_ops.fn_square(x)

        m = Model().to(device=self.device)
        args = (smith.randn(2, 3, device=self.device),)
        with (
            config.patch(
                "aot_inductor.custom_ops_to_c_shims",
                {
                    smith.ops.aoti_custom_ops.fn_square.default: [
                        """
                AOTISmithError
                aoti_smith_cpu_fn_square(
                    AtenTensorHandle input,
                    AtenTensorHandle* ret)""",
                        """
                AOTISmithError
                aoti_smith_cuda_fn_square(
                    AtenTensorHandle input,
                    AtenTensorHandle* ret)""",
                    ],
                },
            ),
            config.patch(
                "aot_inductor.custom_op_libs",
                ["aoti_custom_ops"],
            ),
        ):
            self.check_model(m, args)


class AOTInductorLoggingTest(LoggingTestCase):
    @make_logging_test(dynamic=logging.DEBUG)
    def test_shape_env_reuse(self, records):
        # make sure ShapeEnv is only created once and reused afterwards
        class Foo(smith.nn.Module):
            def forward(self, x):
                return x + 2

        inputs = (smith.randn(4, 4),)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = export(Foo(), inputs, dynamic_shapes=dynamic_shapes, strict=False)
        with smith.no_grad():
            smith._inductor.aot_compile(ep.module(), inputs)
        self.assertEqual([r.msg == "create_env" for r in records].count(True), 1)


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


class AOTICustomOpTestCase(TestCase):
    def setUp(self):
        if IS_SANDCASTLE or IS_FBCODE:
            smith.ops.load_library("//caffe2/test/inductor:custom_ops")
        elif IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        else:
            lib_file_path = find_library_location("libaoti_custom_ops.so")
            if IS_WINDOWS:
                lib_file_path = find_library_location("aoti_custom_ops.dll")
            if not os.path.exists(lib_file_path):
                raise unittest.SkipTest("libaoti_custom_ops not built!")
            smith.ops.load_library(str(lib_file_path))
        super().setUp()


def fail_cpu(is_skip=False):
    return TestFailure(
        ("cpu",),
        is_skip=is_skip,
    )


def fail_gpu(suffixes: tuple[str, ...], is_skip=False):
    return TestFailure(
        suffixes,
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
CPU_TEST_FAILURES = {
    # TODO: failed internally
    "test_multiple_output_alias": fail_cpu(is_skip=True),
}

# test_failures, xfail by default, set is_skip=True to skip
GPU_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quantized_linear": fail_gpu(("cuda", "xpu")),
    "test_quanatized_int8_linear": fail_gpu(("cuda", "xpu")),
    "test_quantized_linear_bias_none": fail_gpu(("cuda", "xpu")),
}


class AOTInductorTestABICompatibleCpu(AOTICustomOpTestCase):
    device = "cpu"
    device_type = "cpu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleCpu,
    "cpu",
    CPU_TEST_FAILURES,
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleGpu(AOTICustomOpTestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleGpu,
    GPU_TYPE,
    GPU_TEST_FAILURES,
)

if __name__ == "__main__":
    from smith._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if HAS_GPU_AND_TRITON or sys.platform == "darwin":
        run_tests(needs="filelock")
