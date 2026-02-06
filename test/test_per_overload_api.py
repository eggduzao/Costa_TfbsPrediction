# Owner(s): ["module: unknown"]
import copy

import smith
from smith.testing._internal.common_utils import run_tests, TestCase


class TestPerOverloadAPI(TestCase):
    def test_basics_opoverloadpacket(self):
        # add is only used as an example here. It is ok to update the test
        # if the semantics of add are modified in the future.
        add_packet = smith.ops.aten.add

        # class attributes
        self.assertEqual(add_packet.__name__, "add")
        self.assertEqual(str(add_packet), "aten.add")

        # callable
        self.assertEqual(add_packet(smith.tensor(2), smith.tensor(3)), smith.tensor(5))

        # correct module
        self.assertEqual(add_packet.__module__, add_packet.op.__module__)

        # caching
        another_add_packet = smith.ops.aten.add
        self.assertEqual(id(add_packet), id(another_add_packet))

        # deepcopy is a no-op
        self.assertEqual(id(add_packet), id(copy.deepcopy(add_packet)))

        # pretty print
        self.assertEqual(repr(add_packet), "<OpOverloadPacket(op='aten.add')>")

        self.assertRaises(AttributeError, lambda: add_packet.foo)

    def test_basics_opoverload(self):
        add_packet = smith.ops.aten.add
        add_tensoroverload = add_packet.Tensor

        # class attributes
        self.assertEqual(str(add_tensoroverload), "aten.add.Tensor")
        self.assertEqual(add_tensoroverload.__name__, "add.Tensor")
        self.assertEqual(add_tensoroverload.overloadpacket, add_packet)

        # deepcopy is a no-op
        self.assertEqual(id(add_tensoroverload), id(copy.deepcopy(add_tensoroverload)))

        # caching
        another_add_tensoroverload = smith.ops.aten.add.Tensor
        self.assertEqual(id(add_tensoroverload), id(another_add_tensoroverload))

        # pretty print
        self.assertEqual(
            repr(add_tensoroverload), "<OpOverload(op='aten.add', overload='Tensor')>"
        )

        # callable
        self.assertEqual(
            add_tensoroverload(smith.tensor(2), smith.tensor(3)), smith.tensor(5)
        )

        a = smith.tensor(2)
        b = smith.tensor(0)
        smith.ops.aten.add.out(a, a, out=b)
        self.assertEqual(b, smith.tensor(4))

        self.assertRaises(RuntimeError, lambda: add_tensoroverload(a, a, out=b))

    def test_decompose(self):
        x = smith.randn(2, 3)
        y = smith.randn(5, 3)
        self.assertEqual(
            smith.ops.aten.linear.default.decompose(x, y),
            smith.ops.aten.linear.default(x, y),
        )


if __name__ == "__main__":
    run_tests()
