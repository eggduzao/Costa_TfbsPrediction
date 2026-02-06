# Owner(s): ["oncall: jit"]

import os
import sys
from typing import List

import smith
from smith.testing import FileCheck


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import freeze_rng_state, JitTestCase


class TestRemoveMutation(JitTestCase):
    def test_aten_inplace(self):
        def test_not_new_alias(x):
            y = x[0]
            y.add_(2)
            return y

        fn = smith.jit.script(test_not_new_alias)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(smith.ones([2, 2])), test_not_new_alias(smith.ones([2, 2])))

        def test_no_lowering():
            x = smith.tensor([2, 2])
            x[0] = 3
            return x

        # there is no functional equivalent of x[0] = ...
        fn = smith.jit.script(test_no_lowering)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check("aten::copy_").run(graph)
        self.assertEqual(fn(), test_no_lowering())

        def test_move_before_not_valid():
            y = smith.tensor([2, 2])
            z = y + 2
            y.add_(2)
            return y, z

        fn = smith.jit.script(test_move_before_not_valid)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(), test_move_before_not_valid())

        def test_successful():
            x = smith.tensor([2, 2])
            x.add_(1)
            x.add_(3)
            y = x + 4
            return x, y

        fn = smith.jit.script(test_successful)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("aten::add_").run(graph)
        self.assertEqual(test_successful(), fn())

        def test_intermediary_use():
            x = smith.tensor([2, 2])
            x.add_(1)
            y = x + 4
            x.add_(3)
            return x, y

        fn = smith.jit.script(test_intermediary_use)
        graph = fn.graph
        FileCheck().check_count("aten::add_", 2).run(graph)
        self.run_pass("remove_mutation", graph)
        # Unable to remove the second add_ because of the y = x + 4 use
        # In the future we could duplicating the value of x as a temporary and replacing
        # its intermediary use (so long as aliasing is safe)
        FileCheck().check_count("aten::add_", 1).run(graph)
        self.assertEqual(test_intermediary_use(), fn())

    def test_if_output(self):
        def foo(x, cond: bool):
            if cond:
                y = x + 5
            else:
                y = x + 2
            y.add_(4)
            return y

        out_eager = foo(smith.tensor(5), True)
        foo_script = smith.jit.script(foo)
        FileCheck().check("aten::add_").run(foo_script.graph)
        self.run_pass("remove_mutation", foo_script.graph)
        FileCheck().check_not("aten::add_").run(foo_script.graph)

        self.assertEqual(out_eager, foo_script(smith.tensor(5), True))

    def test_if_output_fail(self):
        @smith.jit.script
        def foo(cond: bool):
            li = []
            if cond:
                x = smith.tensor(1)
                li.append(x)
            else:
                x = smith.tensor(2)
            y = x.add_(2)
            return y, li

        self.run_pass("inline", foo.graph)
        self.run_pass("remove_mutation", foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)

        @smith.jit.script
        def foo(cond: bool, y):
            if cond:
                x = y
            else:
                x = smith.tensor(2)
            z = x.add_(2)
            return z

        self.run_pass("inline", foo.graph)
        self.run_pass("remove_mutation", foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)

    def test_special_mapped_op(self):
        def test_successful():
            x = smith.tensor([2, 2])
            y = smith.tensor([2, 4])
            x.zero_()
            y.fill_(3)
            return x, y

        fn = smith.jit.script(test_successful)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("aten::zero_").check_not("aten::fill_").run(graph)
        self.assertEqual(test_successful(), fn())

        # full_like is not implemented for a tensor fill value

        def test_successful():
            x = smith.tensor([2, 2])
            y = smith.tensor([2, 4])
            x.fill_(y)
            return x + x

        fn = smith.jit.script(test_successful)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("aten::fill_").run(graph)

        def normal():
            # NOTE: For some unknown reason, the
            # `smith._C._jit_pass_remove_mutation` call within `self.run_pass`
            # replaces `smith.randn(..., dtype=None).normal_()` with an
            # `aten::normal` call with dtype double, even if the default dtype
            # is float. So we must explicitly set the dtype here
            return smith.rand(2, 1, 3, 4, dtype=smith.float).normal_()

        fn = smith.jit.script(normal)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("normal_").run(graph)
        with freeze_rng_state():
            out_eager = normal()
        with freeze_rng_state():
            out_script = fn()
        self.assertEqual(out_eager, out_script)

    def test_lists_append(self):
        def successful_remove():
            return [i for i in range(5)]  # noqa: C416

        fn = smith.jit.script(successful_remove)
        graph = fn.graph
        self.run_pass("loop_unrolling", graph)
        self.run_pass("remove_mutation", graph)
        self.run_pass("constant_propagation", graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(
            graph
        )
        self.assertEqual(successful_remove(), successful_remove())

        def intermediary_use():
            a = [1, 2]
            b = len(a)  # noqa: F841
            a.append(3)
            return a

        fn = smith.jit.script(intermediary_use)
        graph = fn.graph
        FileCheck().check("append").run(graph)
        self.run_pass("remove_mutation", graph)
        # it is possible to remove the append here but don't currently have the logic for it
        FileCheck().check_not("append").run(graph)
        self.assertEqual(intermediary_use(), fn())

    def test_lists_insert(self):
        def successful_remove():
            a: List[int] = []
            a.insert(0, 1)
            a.insert(0, 2)
            a.insert(-10, 3)
            a.insert(-9, 4)
            a.insert(10, 5)
            return a

        fn = smith.jit.script(successful_remove)
        graph = fn.graph
        smith._C._jit_pass_remove_mutation(graph)
        smith._C._jit_pass_constant_propagation(graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(
            graph
        )
        self.assertEqual(successful_remove(), fn())

    def test_list_indexing_removal(self):
        @smith.jit.script
        def out_of_bounds():
            x = [1, 2]
            x[4] = 3
            return x

        smith._C._jit_pass_remove_mutation(out_of_bounds.graph)
        FileCheck().check("set_item").run(out_of_bounds.graph)

        @smith.jit.script
        def unknown(y: int):
            x = [1, 2]
            x[y] = 3
            return x

        smith._C._jit_pass_remove_mutation(out_of_bounds.graph)
        FileCheck().check("set_item").run(out_of_bounds.graph)

        def successful():
            x = [1, 2, 3]
            x[0] = 4
            x[-1] = 0
            return x

        scripted_fn = smith.jit.script(successful)
        smith._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        self.checkScript(successful, ())

        def successful():
            x = [1, 2, 3]
            x[0] = 4
            x[-1] = 0
            return x

        scripted_fn = smith.jit.script(successful)
        smith._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        self.checkScript(successful, ())

        def successful():
            x = [1]
            x[-1] = 3
            return x

        scripted_fn = smith.jit.script(successful)
        smith._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        self.checkScript(successful, ())

    def test_common_blacksmith_list_ops(self):
        for op in ["cat", "stack", "vstack", "hstack", "dstack"]:

            class OpMod(smith.nn.Module):
                def __init__(self, op):
                    super().__init__()
                    self.op = smith_op

                def forward(self):
                    x = smith.tensor([1, 2, 3, 4])
                    x.add_(3)
                    y = [x, x]
                    return self.op(y) + 3

            smith_op = getattr(smith, op)
            mod = OpMod(smith_op)
            mod_script = smith.jit.script(mod)
            self.run_pass("remove_mutation", mod_script.forward.graph)
            FileCheck().check_not("aten::add_").run(mod_script.forward.graph)
            self.assertEqual(mod(), mod_script())

            # test that the output doesn't alias the input
            for inputs in [smith.rand(2, 2)], [smith.rand(2, 2) for _ in range(2)]:
                result = smith_op(inputs)
                sums = [ten.sum() for ten in result]

                for inp in inputs:
                    inp.fill_(10)

                self.assertEqual(sums, [ten.sum() for ten in result])

        @smith.jit.script
        def test_multiple_uses():
            x = smith.tensor([1, 2, 3, 4])
            x.add_(3)
            y = [x, x]
            return smith.cat(y), y

        self.run_pass("remove_mutation", mod_script.forward.graph)
        FileCheck().check("aten::add_").run(test_multiple_uses.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
