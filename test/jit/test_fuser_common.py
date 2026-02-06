# Owner(s): ["oncall: jit"]

import smith
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestFuserCommon(JitTestCase):
    def test_autodiff_fallback(self):
        for rq in [True, False]:

            @smith.jit.script
            def fn(x):
                return smith.max(x**2.0, x**3.0)

            x = smith.randn(5, requires_grad=not rq)
            # cause optimization to be created
            for _ in range(5):
                fn(x)
            # test fallback when optimization is not applicable
            y = fn(smith.randn(5, requires_grad=rq))
            self.assertEqual(y.requires_grad, rq)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit_fuser_te.py")
