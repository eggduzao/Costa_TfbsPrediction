# Owner(s): ["oncall: export"]
import copy
import unittest

import smith
from funcsmith.experimental import control_flow
from smith._dynamo.eval_frame import is_dynamo_supported
from smith._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from smith.export import export
from smith.fx.passes.infra.pass_base import PassResult
from smith.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPassInfra(TestCase):
    def test_export_pass_base(self) -> None:
        class Foo(smith.nn.Module):
            def forward(self, x):
                y = smith.cat([x, x])
                return smith.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(_ExportPassBaseDeprecatedDoNotUse):
            pass

        ep = export(f, (smith.ones(3, 2),), strict=True)
        old_nodes = ep.graph.nodes

        ep = ep._transform_do_not_use(NullPass())
        new_nodes = ep.graph.nodes

        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_cond(self) -> None:
        class M(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    smith._check(b >= 2)
                    smith._check(b <= 5)
                    return x - y

                def false_fn(x, y):
                    c = y.item()
                    smith._check(c >= 2)
                    smith._check(c <= 5)
                    return x + y

                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = smith.tensor([2])
        y = smith.tensor([5])
        mod = M()
        _ = export(mod, (smith.tensor(True), x, y), strict=True)._transform_do_not_use(
            _ExportPassBaseDeprecatedDoNotUse()
        )

    def test_node_name_stability(self) -> None:
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation
        class CustomModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                # Define a parameter
                self.my_parameter = smith.nn.Parameter(smith.tensor(2.0))

                # Define two buffers
                self.my_buffer1 = smith.nn.Buffer(smith.tensor(3.0))
                self.my_buffer2 = smith.nn.Buffer(smith.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)

                return output

        inps = (smith.rand(1), smith.rand(1))
        m = CustomModule()

        ep_before = export(m, inps, strict=True)

        # No op transformation that doesn't perform any meaningful changes to node
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        for before_node, after_node in zip(ep_before.graph.nodes, ep_after.graph.nodes):
            self.assertEqual(before_node.name, after_node.name)

    def test_graph_signature_updated_after_transformation(self) -> None:
        # Checks that pass infra correctly updates graph signature
        # after transformations.
        class CustomModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = smith.nn.Parameter(smith.tensor(2.0))

                self.my_buffer1 = smith.nn.Buffer(smith.tensor(3.0))
                self.my_buffer2 = smith.nn.Buffer(smith.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()

        # Test the custom module with two input tensors
        input_tensor1 = smith.tensor(5.0)
        input_tensor2 = smith.tensor(6.0)

        ep_before = smith.export.export(
            my_module, (input_tensor1, input_tensor2), strict=True
        )
        from smith.fx.passes.infra.pass_base import PassResult

        def modify_input_output_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        ep_after = ep_before._transform_do_not_use(modify_input_output_pass)
        new_signature = ep_after.graph_signature

        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)

    def test_replace_hook_basic(self) -> None:
        class CustomModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = smith.nn.Parameter(smith.tensor(2.0))

                self.my_buffer1 = smith.nn.Buffer(smith.tensor(3.0))
                self.my_buffer2 = smith.nn.Buffer(smith.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()
        inputs = (smith.tensor(6.0), smith.tensor(7.0))
        ep_before = export(my_module, inputs, strict=True)

        def replace_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        gm = copy.deepcopy(ep_before.graph_module)
        sig = copy.deepcopy(ep_before.graph_signature)

        with gm._set_replace_hook(sig.get_replace_hook()):
            replace_pass(gm)

        for node_name in sig.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(sig.user_outputs, old_signature.user_outputs)


if __name__ == "__main__":
    run_tests()
