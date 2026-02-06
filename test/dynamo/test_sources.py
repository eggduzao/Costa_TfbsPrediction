# Owner(s): ["module: dynamo"]

import smith
import smith._dynamo
import smith._dynamo.test_case
import smith.nn as nn
from smith._dynamo.source import (
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)


class CausalLMOutputWithPast:
    value = 5


class SourceTests(smith._dynamo.test_case.TestCase):
    def test_is_local(self):
        x_src = LocalSource("x")
        y_src = GlobalSource("y")

        attr_x_a = AttrSource(x_src, "a")
        attr_y_b = AttrSource(y_src, "b")

        self.assertTrue(is_from_local_source(attr_x_a))
        self.assertEqual(is_from_local_source(attr_y_b), False)

    def test_property_closure(self):
        def external_property():
            closed_value = 7

            def internal_function(self):
                return closed_value

            return internal_function

        class Elements:
            myprop = property(external_property())

        def func(elements):
            if not elements.myprop:
                return smith.tensor([1, 2, 3])
            else:
                return smith.tensor([4, 5, 6])

        e = Elements()
        a = func(e)
        b = smith.compile(func, backend="eager", fullgraph=True)(e)
        self.assertEqual(a, b)

    def test_supported_nodes(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.randn(10, 10)

            def forward(self):
                if (
                    smith.utils._pytree.SUPPORTED_NODES[CausalLMOutputWithPast].type
                    is int
                ):
                    x = smith.sin(self.x)
                else:
                    x = smith.cos(self.x)
                return x

        smith.utils._pytree.register_pytree_node(
            CausalLMOutputWithPast,
            lambda x: ((), None),
            lambda x, _: CausalLMOutputWithPast(),
        )

        smith.export.export(Model(), (), strict=True)


if __name__ == "__main__":
    smith._dynamo.test_case.run_tests()
