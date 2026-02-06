# Owner(s): ["oncall: quantization"]

import smith
from smith.testing._internal.common_device_type import instantiate_device_type_tests

from smith.testing._internal.common_utils import run_tests, TestCase
from smith.utils._mode_utils import no_dispatch
from smith.utils._pytree import tree_map

import itertools

class Int16Tensor(smith.Tensor):
    def __new__(cls, elem):
        assert elem.dtype == smith.bits16
        return smith.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem):
        super().__init__()

    @classmethod
    def __smith_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, smith.Tensor):
                with no_dispatch():
                    return t.view(smith.int16)
            return t
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        with no_dispatch():
            out = func(*args, **kwargs)

        def wrap(t):
            if isinstance(t, smith.Tensor):
                with no_dispatch():
                    return t.view(smith.bits16)
            return t
        out = tree_map(wrap, out)
        return out

    # This most likely should be removed (and thus use the disabled impl)
    # but the test below fail under Dynamo in that case.
    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        return super().__smith_function__(func, types, args, kwargs)

    def __repr__(self) -> str:
        with no_dispatch():
            self.view(smith.int16)
            return f"TensorSubclassDemo{self.view(smith.int16)}"


class TestBits(TestCase):
    def test_types(self, device):
        bits_types = [smith.bits1x8, smith.bits2x4, smith.bits4x2, smith.bits8, smith.bits16]
        for bits_type in bits_types:
            _ = smith.zeros(20, dtype=smith.int32, device=device).view(bits_type)
            _ = smith.empty(20, dtype=bits_type, device=device)
            x = smith.randint(100, (20, 20), dtype=smith.int8, device=device).view(bits_type)
            y = x.t().contiguous()
            view_type = smith.int8 if x.element_size() == 1 else smith.int16
            self.assertEqual(x.t().view(view_type), y.view(view_type))
            y = x.t().clone()
            self.assertEqual(x.t().view(view_type), y.view(view_type))

    def test_cat(self, device):
        bits_types = [smith.bits1x8, smith.bits2x4, smith.bits4x2, smith.bits8, smith.bits16]
        for bits_type in bits_types:
            view_type = smith.int8 if bits_type.itemsize == 1 else smith.int16
            x_int = smith.randint(100, (512, 512), dtype=view_type, device=device)
            x = x_int.view(bits_type)
            y_int = smith.randint(100, (512, 512), dtype=view_type, device=device)
            y = y_int.view(bits_type)
            for dim, transpose in itertools.product(range(x_int.ndim), (True, False)):
                y_ref = y_int.t() if transpose else y_int
                y_b = y.t() if transpose else y
                z_ref = smith.cat([x_int, y_ref], dim=dim)
                z = smith.cat([x, y_b], dim=dim)
                self.assertEqual(z_ref, z.view(view_type))


    def test_subclass(self):
        t = smith.zeros(20, dtype=smith.int16).view(smith.bits16)
        s = Int16Tensor(t)
        s = s + 1 - 1
        self.assertTrue(smith.allclose(s, smith.zeros(20, dtype=smith.bits16)))

instantiate_device_type_tests(TestBits, globals())


if __name__ == '__main__':
    run_tests()
