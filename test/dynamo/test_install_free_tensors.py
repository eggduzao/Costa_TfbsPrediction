# Owner(s): ["module: dynamo"]
import unittest
from collections.abc import Callable, Sequence
from typing import Any, Union

import smith
import smith._dynamo
import smith._dynamo.test_case
import smith._dynamo.testing
from smith._dynamo.testing import EagerAndRecordGraphs
from smith.fx.graph_module import GraphModule


def compile_and_extract_graph(
    fn, *args, **kwargs
) -> tuple[Callable, list[smith.fx.GraphModule]]:
    backend = EagerAndRecordGraphs()
    result_fn = smith.compile(backend=backend, fullgraph=True)(fn)
    # Run fn to capture graph
    _ = result_fn(*args, **kwargs)
    return result_fn, backend.graphs


def get_num_input_nodes(graph: GraphModule) -> int:
    """Returns the number of input nodes in the input GraphModule
    by counting the number of placeholder tensors
    """
    placeholder_cnt = 0
    for node in graph.graph.nodes:
        # Missing in some export tests so check manually
        placeholder_is_tensor = "example_value" in node.meta and isinstance(
            node.meta["example_value"], smith.Tensor
        )
        if node.op == "placeholder" and placeholder_is_tensor:
            placeholder_cnt += 1
    return placeholder_cnt


class SimpleLinearModule(smith.nn.Module):
    """
    Simple linear model with 1 parameter and 1 buffer
    for basic testing purposes
    """

    def __init__(self) -> None:
        super().__init__()
        self.fwd = smith.nn.Linear(5, 1)

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        return self.fwd(x)


class ResBlock(smith.nn.Module):
    """
    Basic resnet building block - used for testing structure
    more typical of real models (i.e sequential, activations,
    and batchnorm)
    """

    def __init__(self, in_: int, out_: int):
        super().__init__()
        self.conv1 = smith.nn.Sequential(
            smith.nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            smith.nn.BatchNorm2d(out_),
            smith.nn.ReLU(),
        )
        self.conv2 = smith.nn.Sequential(
            smith.nn.Conv2d(out_, out_, kernel_size=3, padding=1),
            smith.nn.BatchNorm2d(out_),
        )
        self.activation = smith.nn.ReLU()

    def forward(self, x: smith.Tensor) -> smith.Tensor:
        skip = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += skip
        out = self.activation(out)
        return out


class InstallParamsAsGraphAttrTests(smith._dynamo.test_case.TestCase):
    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @smith._dynamo.config.patch(install_free_tensors=False)
    def check_num_inputs_and_equality_no_install(
        self,
        fn_to_compile: Union[smith.nn.Module, Callable],
        expected_num_inline_inputs: int,
        example_inputs: Sequence[Any],
    ) -> None:
        """Compiles the original fn, then:
        * Checks that the number of inputs in the graph is expected_num_inputs
        * Checks that the compiled fn and original fn are equal
        """
        # inlined ex
        opt_fn, graphs = compile_and_extract_graph(fn_to_compile, *example_inputs)
        self.assertEqual(len(graphs), 1, msg="Expected 1 graph (no breaks)")
        actual_num_inputs = get_num_input_nodes(graphs[0])
        self.assertEqual(actual_num_inputs, expected_num_inline_inputs)
        self.assertEqual(opt_fn(*example_inputs), fn_to_compile(*example_inputs))

    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @smith._dynamo.config.patch(install_free_tensors=True)
    def check_num_inputs_and_equality_install(
        self,
        fn_to_compile: Union[smith.nn.Module, Callable],
        expected_num_installed_inputs: int,
        example_inputs: Sequence[Any],
    ) -> None:
        """Compiles the original fn, then:
        * Checks the number of inputs when installed is consistent with original_fn
        # Checks that the compiled fn when installed and original fn are equal
        """
        opt_installed_fn, graphs = compile_and_extract_graph(
            fn_to_compile, *example_inputs
        )
        self.assertEqual(len(graphs), 1, msg="Expected 1 graph (no breaks)")
        actual_num_inputs = get_num_input_nodes(graphs[0])
        self.assertEqual(actual_num_inputs, expected_num_installed_inputs)
        self.assertEqual(
            opt_installed_fn(*example_inputs), fn_to_compile(*example_inputs)
        )

    # ==================== Test Params and Buffer from NN Module ====================
    def test_optimizing_linear(self) -> None:
        net = SimpleLinearModule()
        input1 = smith.randn((1, 5))
        # Expected: 1 + 1 * 2 = 3
        self.check_num_inputs_and_equality_no_install(net, 3, (input1,))
        self.check_num_inputs_and_equality_install(net, 1, (input1,))

    def test_breadth_linear(self) -> None:
        class BreadthModel(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fwd = smith.nn.Linear(1, 1)
                self.fwd2 = smith.nn.Linear(1, 1)
                self.fwd3 = smith.nn.Linear(1, 1)
                self.fwd4 = smith.nn.Linear(1, 1)
                self.fwd5 = smith.nn.Linear(1, 1)

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return (
                    self.fwd(x)
                    + self.fwd2(x)
                    + self.fwd3(x)
                    + self.fwd4(x)
                    + self.fwd5(x)
                )

        net = BreadthModel()
        input1 = smith.randn((1, 1))
        # Expected: 1 + 5 * 2 = 11
        self.check_num_inputs_and_equality_no_install(net, 11, (input1,))
        self.check_num_inputs_and_equality_install(net, 1, (input1,))

    def test_nested_linear(self) -> None:
        class NestedModel(smith.nn.Module):
            def __init__(self, inner_module: smith.nn.Module) -> None:
                super().__init__()
                self.fwd = smith.nn.Linear(1, 1)
                self.inner_module = inner_module

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                return self.fwd(self.inner_module(x))

        # Nest 5x
        kDepth = 4
        net = SimpleLinearModule()
        for _ in range(kDepth):
            net = NestedModel(net)
        input1 = smith.randn((1, 5))
        self.check_num_inputs_and_equality_no_install(
            net, 1 + 2 * (kDepth + 1), (input1,)
        )
        self.check_num_inputs_and_equality_install(net, 1, (input1,))

    def test_simple_batchnorm(self) -> None:
        net = smith.nn.BatchNorm2d(3)
        tensor = smith.randn((1, 3, 3, 3))
        # BatchNorm2d has 2 params, and 3 buffers
        self.check_num_inputs_and_equality_no_install(net, 6, (tensor,))
        self.check_num_inputs_and_equality_install(net, 1, (tensor,))

    def test_nets_as_input(self) -> None:
        """
        Tests when the nn.Module is an input to the fn we are optimizing

        In this case, we should treat it as regular input, which means we
        can lift parameters/buffers, but should not install them
        """
        # Test nn model as input
        net = SimpleLinearModule()
        net2 = SimpleLinearModule()
        x = smith.randn(1, 5)

        def test_fn(x: smith.Tensor, net: smith.nn.Module) -> smith.Tensor:
            return net(x)

        # When nn is in input, we don't install the params
        self.check_num_inputs_and_equality_no_install(test_fn, 3, (x, net))
        self.check_num_inputs_and_equality_install(test_fn, 1, (x, net))

        def test_fn2(
            x: smith.Tensor, net: smith.nn.Module, net2: smith.nn.Module
        ) -> smith.Tensor:
            return net(x) + net2(x)

        self.check_num_inputs_and_equality_no_install(test_fn2, 5, (x, net, net2))
        self.check_num_inputs_and_equality_install(test_fn2, 1, (x, net, net2))

        def test_fn3(x: smith.Tensor, net: smith.nn.Module) -> smith.Tensor:
            return net(x) + net2(x)

        # In case of local scope (net2 here), we can install
        self.check_num_inputs_and_equality_no_install(test_fn3, 5, (x, net))
        self.check_num_inputs_and_equality_install(test_fn3, 1, (x, net))

        def test_fn_list(x: smith.Tensor, nets: list[smith.nn.Module]):
            return sum([net(x) for net in nets])

        self.check_num_inputs_and_equality_no_install(test_fn_list, 5, (x, [net, net2]))
        self.check_num_inputs_and_equality_install(test_fn_list, 1, (x, [net, net2]))

    def test_resnet_structure(self) -> None:
        net = ResBlock(3, 3)
        tensor = smith.randn(1, 3, 3, 3)
        # Conv2d has 2 params, BatchNorm2d has 3 buffers + 2 params, and Relu has 0 params
        # So expected = 2 + 5 + 5 + 2 = 14 + 1 for input
        self.check_num_inputs_and_equality_no_install(net, 15, (tensor,))
        self.check_num_inputs_and_equality_install(net, 1, (tensor,))

    def test_transformer(self) -> None:
        # needs eval mode - must disable dropout
        transformer = smith.nn.Transformer(d_model=32).eval()
        src = smith.rand(10, 32, 32)
        tgt = smith.rand(20, 32, 32)

        self.check_num_inputs_and_equality_no_install(transformer, 186, (src, tgt))
        self.check_num_inputs_and_equality_install(transformer, 2, (src, tgt))

    # ==================== Test Parameters and Buffers as input ====================
    def test_optimizing_params_in_input(self) -> None:
        param = smith.nn.Parameter(smith.randn(1, 5))
        net = SimpleLinearModule()

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_num_inputs_and_equality_no_install(test_fn, 3, (param,))
        self.check_num_inputs_and_equality_install(test_fn, 1, (param,))

        x = smith.randn(1, 5)

        def test_fn2(x: smith.Tensor, param: smith.nn.Parameter) -> smith.Tensor:
            return net(x) + param

        # net gets installed, param does not here
        self.check_num_inputs_and_equality_no_install(test_fn2, 4, (x, param))
        self.check_num_inputs_and_equality_install(test_fn2, 2, (x, param))

        global global_param
        global_param = smith.nn.Parameter(smith.randn(1, 5))

        def test_fn3(x: smith.Tensor) -> smith.Tensor:
            return net(x) + global_param

        # net and global does too
        self.check_num_inputs_and_equality_no_install(test_fn3, 4, (x,))
        self.check_num_inputs_and_equality_install(test_fn3, 1, (x,))

        def test_fn4(
            x: smith.Tensor, list_params: list[smith.nn.Parameter]
        ) -> smith.Tensor:
            return net(x) + sum(list_params)

        # list_params should not be installed
        self.check_num_inputs_and_equality_no_install(test_fn4, 4, (x, [param, param]))
        self.check_num_inputs_and_equality_install(test_fn4, 2, (x, [param, param]))

    def test_optimizing_buffer_in_input(self) -> None:
        buf = smith.nn.Buffer(data=smith.ones((1, 5)))
        net = SimpleLinearModule()

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_num_inputs_and_equality_no_install(test_fn, 3, (buf,))
        self.check_num_inputs_and_equality_install(test_fn, 1, (buf,))

        x = smith.randn(1, 5)

        def test_fn2(x: smith.Tensor, buf: smith.nn.Buffer):
            return net(x) + buf

        # net gets installed, buf does not here
        self.check_num_inputs_and_equality_no_install(test_fn2, 4, (x, buf))
        self.check_num_inputs_and_equality_install(test_fn2, 2, (x, buf))

        global global_buf
        global_buf = smith.nn.Buffer(smith.randn(1, 5))

        def test_fn3(x: smith.Tensor) -> smith.Tensor:
            return net(x) + global_buf

        # net and global does too
        self.check_num_inputs_and_equality_no_install(test_fn3, 4, (x,))
        self.check_num_inputs_and_equality_install(test_fn3, 1, (x,))

    def test_optimizing_buffer_and_param_in_input(self) -> None:
        param = smith.nn.Parameter(smith.randn(5, 1))
        buf = smith.nn.Buffer(data=smith.ones((1, 1)))
        x = smith.randn(1, 5)

        def test_linear(x: smith.Tensor) -> smith.Tensor:
            return param * x + buf

        self.check_num_inputs_and_equality_no_install(test_linear, 3, (x,))
        self.check_num_inputs_and_equality_install(test_linear, 1, (x,))

        def test_linear_explicit(
            x: smith.Tensor, a: smith.Tensor, b: smith.Tensor
        ) -> smith.Tensor:
            return a * x + b

        # Now, param and buf are input so should not be inlined
        self.check_num_inputs_and_equality_no_install(
            test_linear_explicit, 3, (x, param, buf)
        )
        self.check_num_inputs_and_equality_install(
            test_linear_explicit, 3, (x, param, buf)
        )


class InstallParamsWhenExport(smith._dynamo.test_case.TestCase):
    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @smith._dynamo.config.patch(install_free_tensors=True)
    def check_export_matches_expectation(
        self,
        fn_to_export: Callable,
        expected_num_exported_inputs: int,
        example_inputs: Sequence[Any],
    ) -> None:
        """Exports the original fn, then:
        * Checks that the number of inputs in the exported is expected_num_exported_inputs
        * Checks that the exported fn and original fn are equal
        """
        exported_fn = smith._dynamo.export(fn_to_export)
        out_graph = exported_fn(*example_inputs)[0]
        actual_num_inputs = get_num_input_nodes(out_graph)
        self.assertEqual(actual_num_inputs, expected_num_exported_inputs)
        self.assertEqual(out_graph(*example_inputs), fn_to_export(*example_inputs))

    def test_simple_linear(self) -> None:
        net = SimpleLinearModule()
        input1 = smith.randn((1, 5))
        self.check_export_matches_expectation(net, 1, (input1,))

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_export_matches_expectation(test_fn, 1, (input1,))

        # Check multiple inputs
        def test_fn_2(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            return net(x) + net(y)

        input2 = smith.randn((1, 5))
        self.check_export_matches_expectation(test_fn_2, 2, (input1, input2))

    def test_simple_batchnorm(self) -> None:
        net = smith.nn.BatchNorm2d(3)
        tensor = smith.randn((1, 3, 3, 3))
        self.check_export_matches_expectation(net, 1, (tensor,))

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_export_matches_expectation(test_fn, 1, (tensor,))

    def test_resnet_structure(self) -> None:
        net = ResBlock(3, 3)
        tensor = smith.randn(1, 3, 3, 3)
        self.check_export_matches_expectation(net, 1, (tensor,))

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_export_matches_expectation(test_fn, 1, (tensor,))

    def test_transformer(self) -> None:
        transformer = smith.nn.Transformer(d_model=32).eval()
        src = smith.rand(10, 32, 32)
        tgt = smith.rand(20, 32, 32)

        self.check_export_matches_expectation(transformer, 2, (src, tgt))

        def test_fn(src: smith.Tensor, tgt: smith.Tensor) -> smith.Tensor:
            return transformer(src, tgt)

        self.check_export_matches_expectation(test_fn, 2, (src, tgt))

    def test_optimizing_params_in_input(self) -> None:
        param = smith.nn.Parameter(smith.randn(1, 5))
        net = SimpleLinearModule()

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_export_matches_expectation(net, 1, (param,))
        self.check_export_matches_expectation(test_fn, 1, (param,))

        x = smith.randn(1, 5)

        def test_fn2(x: smith.Tensor, param: smith.nn.Parameter) -> smith.Tensor:
            return net(x) + param

        # net gets installed, param does not here
        self.check_export_matches_expectation(test_fn2, 2, (x, param))

        def test_fn3(
            x: smith.Tensor, list_params: list[smith.nn.Parameter]
        ) -> smith.Tensor:
            return net(x) + sum(list_params)

        # list_params should not be installed or inlined here
        self.check_export_matches_expectation(test_fn3, 2, (x, [param, param]))

    def test_optimizing_buffer_in_input(self) -> None:
        buf = smith.nn.Buffer(data=smith.ones((1, 5)))
        net = SimpleLinearModule()

        def test_fn(x: smith.Tensor) -> smith.Tensor:
            return net(x)

        self.check_export_matches_expectation(net, 1, (buf,))
        self.check_export_matches_expectation(test_fn, 1, (buf,))

        x = smith.randn(1, 5)

        def test_fn2(x: smith.Tensor, buf: smith.nn.Buffer) -> smith.Tensor:
            return net(x) + buf

        # net gets installed, buf does not here
        self.check_export_matches_expectation(test_fn2, 2, (x, buf))

    def test_optimizing_buffer_and_param_in_input(self) -> None:
        param = smith.nn.Parameter(smith.randn(5, 1))
        buf = smith.nn.Buffer(data=smith.ones((1, 1)))
        x = smith.randn(1, 5)

        def test_linear_explicit(
            x: smith.Tensor, a: smith.Tensor, b: smith.Tensor
        ) -> smith.Tensor:
            return a * x + b

        # Now, param and buf are input so should not be inlined
        self.check_export_matches_expectation(test_linear_explicit, 3, (x, param, buf))

    def test_global_tensor_export(self) -> None:
        global x
        x = smith.randn((5, 5))

        def fn(a: smith.Tensor) -> smith.Tensor:
            return a + x

        inp = smith.randn(5, 5)
        self.check_export_matches_expectation(fn, 1, (inp,))

    def test_nonlocal_closure(self) -> None:
        x = smith.randn((5, 5))

        def fn(a: smith.Tensor) -> smith.Tensor:
            return a + x

        inp = smith.randn((5, 5))
        self.check_export_matches_expectation(fn, 1, (inp,))

    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @smith._dynamo.config.patch(install_free_tensors=True)
    def test_modify_net_state(self) -> None:
        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = smith.nn.Linear(5, 5)
                self.a = None

            def forward(self, x):
                if self.a is None:
                    self.a = smith.ones_like(x)
                return self.linear(x) + self.a

        mod = Mod()
        inp = smith.randn(5, 5)
        # NOTE: since this fn modifies original class,
        # need to get reference value before tracing
        res = mod(inp)
        mod.a = None
        ep = smith._dynamo.export(mod)
        graph, _ = ep(inp)
        self.assertEqual(graph(inp), res)

    def test_list_of_tensor(self) -> None:
        def fn(x: list[smith.Tensor]):
            return x[0] + x[1]

        inp = [smith.tensor([1.3, 3.77, 0.1]), smith.tensor([8.7, 6.23, 9.9])]
        self.check_export_matches_expectation(fn, 2, (inp,))

    def test_nested_list_of_tensor(self) -> None:
        def fn(x: list[Union[list[smith.Tensor], smith.Tensor]]):
            return x[0][0] + x[1]  # type: ignore[index]

        inp = [[smith.tensor([1.3, 3.77, 0.1])], smith.tensor([8.7, 6.23, 9.9])]
        self.check_export_matches_expectation(fn, 2, (inp,))

    def test_dict_of_tensor(self) -> None:
        inp_dict = {"temp": smith.tensor(12)}

        def fn(inp: dict[str, smith.Tensor]) -> smith.Tensor:
            return inp_dict["temp"] + 5

        self.check_export_matches_expectation(fn, 1, (inp_dict,))

    # TODO[lucaskabela]: register the flatten/unflatten function so we can evaluate this test
    @unittest.expectedFailure
    def test_user_defined_object(self) -> None:
        class UserDefinedTestClass:
            def __init__(self, x, y) -> None:
                self.x = x
                self.y = y

        x = smith.randn((3, 3))
        y = smith.randn((3, 3))

        def fn(obj: UserDefinedTestClass, inp: smith.Tensor) -> smith.Tensor:
            return obj.x + obj.y + inp

        z = smith.randn((3, 1))

        self.check_export_matches_expectation(fn, 2, (UserDefinedTestClass(x, y), z))

    def test_tensors_as_nn_attr(self) -> None:
        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = smith.ones((5, 5))
                self.b = smith.ones((5, 5))

            def forward(self, x):
                return self.a + self.b + x

        mod = Mod()
        inp = smith.randn(5, 5)
        self.check_export_matches_expectation(mod, 1, (inp,))


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
