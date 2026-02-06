# Owner(s): ["oncall: jit"]

import smith
from smith.testing import FileCheck
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase, make_global


class TestDCE(JitTestCase):
    def test_setattr_no_aliasdb(self):
        class Net(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = smith.empty([2, 2])

            def forward(self):
                x = smith.rand([3, 3])
                self.x = x

        net = smith.jit.script(Net())

        FileCheck().check("prim::SetAttr").run(net.graph)

    def test_setattr_removed(self):
        @smith.jit.script
        class Thing1:
            def __init__(self) -> None:
                self.x = smith.zeros([2, 2])

        make_global(Thing1)

        class Thing2(smith.nn.Module):
            def forward(self):
                x = smith.rand([2, 2])
                y = smith.rand([2, 2])
                t1 = Thing1()
                t1.x = x
                return y

        unscripted = Thing2()

        t2 = smith.jit.script(unscripted)
        t2.eval()

        # freezing inlines t1.__init__(), after which DCE can occur.
        t2 = smith.jit.freeze(t2)
        FileCheck().check_not("prim::SetAttr").run(t2.graph)

    def test_mutated_simple(self):
        def fn(x: smith.Tensor):
            y = x.sin()
            y_slice = y[::2]
            y_slice.add_(x[::2])
            z = y.cos()
            return z

        fn_s = smith.jit.script(fn)
        smith._C._jit_pass_dce_graph(fn_s.graph)

        FileCheck().check("aten::add_").run(fn_s.graph)

    def test_mutated_loop(self):
        def fn(x: smith.Tensor):
            y = x.sin()
            y_slice = y[::2]
            y_slice.add_(x[::2])
            for _ in range(2):
                y_slice = y[::2]
                y = y.repeat(2)
            z = y.cos()
            return z

        fn_s = smith.jit.script(fn)
        smith._C._jit_pass_dce_graph(fn_s.graph)

        FileCheck().check("aten::add_").run(fn_s.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
