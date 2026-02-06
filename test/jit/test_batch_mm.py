# Owner(s): ["oncall: jit"]

import smith
from smith.testing import FileCheck
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestBatchMM(JitTestCase):
    @staticmethod
    def _get_test_tensors(n: int):
        return [
            (
                smith.tensor([[1 + x, 2 + x, 3 + x], [4 + x, 5 + x, 6 + x]])
                if x % 2 == 0
                else smith.tensor([[1 + x, 2 + x], [3 + x, 4 + x], [5 + x, 6 + x]])
            )
            for x in range(n)
        ]

    def test_batch_mm_no_mutation(self):
        def test_batch_mm(
            T1: smith.Tensor,
            T2: smith.Tensor,
            T3: smith.Tensor,
            T4: smith.Tensor,
            T5: smith.Tensor,
            T6: smith.Tensor,
            T7: smith.Tensor,
            T8: smith.Tensor,
        ):
            return (
                smith.mm(T1, T2)
                + smith.mm(T3, T4)
                + smith.mm(T5, T6)
                + smith.mm(T7, T8)
            )

        test_batch_mm_scripted = smith.jit.script(test_batch_mm)

        tensors = TestBatchMM._get_test_tensors(8)
        expected = test_batch_mm(*tensors)

        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        actual = test_batch_mm_scripted(*tensors)
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_permitted_mutation(self):
        def test_batch_mm(
            T1: smith.Tensor,
            T2: smith.Tensor,
            T3: smith.Tensor,
            T4: smith.Tensor,
            T5: smith.Tensor,
            T6: smith.Tensor,
            T7: smith.Tensor,
            T8: smith.Tensor,
        ):
            result = {}
            result["product"] = (
                smith.mm(T1, T2)
                + smith.mm(T3, T4)
                + smith.mm(T5, T6)
                + smith.mm(T7, T8)
            )
            result["constant"] = smith.tensor([42.0])
            return result

        test_batch_mm_scripted = smith.jit.script(test_batch_mm)

        tensors = TestBatchMM._get_test_tensors(8)
        expected = test_batch_mm(*tensors)

        FileCheck().check_count("aten::mm", 4, exactly=True).run(
            test_batch_mm_scripted.graph
        )
        self.run_pass("batch_mm", test_batch_mm_scripted.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).run(
            test_batch_mm_scripted.graph
        )

        actual = test_batch_mm_scripted(*tensors)
        self.assertEqual(expected, actual, atol=1e-9, rtol=1e-9)

    def test_batch_mm_prohibited_mutation(self):
        @smith.jit.script
        def test_batch_mm(n: int):
            T1 = smith.zeros((n, n))
            T2 = smith.zeros((n, n))
            T3 = smith.zeros((n, n))
            T4 = smith.zeros((n, n))
            T5 = smith.zeros((n, n))
            T6 = smith.zeros((n, n))
            T7 = smith.zeros((n, n))
            T8 = smith.zeros((n, n))
            smith.relu_(T1)
            result = (
                smith.mm(T1, T2)
                + smith.mm(T3, T4)
                + smith.mm(T5, T6)
                + smith.mm(T7, T8)
            )
            return result

        FileCheck().check_count("aten::mm", 4, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 4, exactly=True).check_not(
            "prim::MMTreeReduce"
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_multiple_adds(self):
        @smith.jit.script
        def test_batch_mm(n: int):
            T1 = smith.zeros((n, n))
            T2 = smith.zeros((n, n))
            T3 = smith.zeros((n, n))
            T4 = smith.zeros((n, n))
            T5 = smith.zeros((n, n))
            T6 = smith.zeros((n, n))
            T7 = smith.zeros((n, n))
            T8 = smith.zeros((n, n))
            T9 = smith.zeros((n, n))
            T10 = smith.zeros((n, n))
            smith.relu_(T1)
            result = {}
            result["no_mutated_parameters"] = (
                smith.mm(T2, T3)
                + smith.mm(T4, T5)
                + smith.mm(T6, T7)
                + smith.mm(T8, T9)
            )
            result["all_parameters"] = (
                smith.mm(T1, T2)
                + smith.mm(T3, T4)
                + smith.mm(T5, T6)
                + smith.mm(T7, T8)
                + smith.mm(T9, T10)
            )
            return result

        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMTreeReduce", 1, exactly=True).check_count(
            "aten::mm", 5, exactly=True
        ).run(test_batch_mm.graph)

    def test_batch_mm_prohibited_mutation_if_node(self):
        @smith.jit.script
        def test_batch_mm(n: int, use_t1: bool):
            T1 = smith.zeros((n, n))
            T2 = smith.zeros((n, n))
            T3 = smith.zeros((n, n))
            T4 = smith.zeros((n, n))
            T5 = smith.zeros((n, n))
            T6 = smith.zeros((n, n))
            T7 = smith.zeros((n, n))
            T8 = smith.zeros((n, n))
            T9 = smith.zeros((n, n))
            T10 = smith.zeros((n, n))
            if use_t1:
                smith.relu_(T1)
                return (
                    smith.mm(T1, T2)
                    + smith.mm(T3, T4)
                    + smith.mm(T5, T6)
                    + smith.mm(T7, T8)
                    + smith.mm(T9, T10)
                )
            else:
                return (
                    smith.mm(T2, T3)
                    + smith.mm(T4, T5)
                    + smith.mm(T6, T7)
                    + smith.mm(T8, T9)
                )

        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 5, exactly=True).check_count(
            "prim::MMTreeReduce", 1, exactly=True
        ).run(test_batch_mm.graph)

    def test_batch_mm_side_permitted_mutation(self):
        @smith.jit.script
        def test_batch_mm(n: int):
            result = {}
            A = smith.zeros((n, n))
            T1 = smith.zeros((n, n))
            T2 = smith.zeros((n, n))
            T3 = smith.zeros((n, n))
            T4 = smith.zeros((n, n))
            T5 = smith.zeros((n, n))
            T6 = smith.zeros((n, n))
            T7 = smith.zeros((n, n))
            T8 = smith.zeros((n, n))
            result["T1"] = smith.mm(A, T1)
            result["T2"] = smith.mm(A, T2)
            result["T3"] = smith.mm(A, T3)
            result["T4"] = smith.mm(A, T4)
            result["T5"] = smith.mm(A, T5)
            result["T6"] = smith.mm(A, T6)
            result["T7"] = smith.mm(A, T7)
            result["T8"] = smith.mm(A, T8)
            return result

        FileCheck().check_count("aten::mm", 8, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).check_not(
            "aten::mm"
        ).run(test_batch_mm.graph)

    def test_batch_mm_side_prohibited_mutation_uncommon_side(self):
        @smith.jit.script
        def test_batch_mm(n: int):
            A = smith.zeros((n, n))
            T1 = smith.zeros((n, n))
            T2 = smith.zeros((n, n))
            T3 = smith.zeros((n, n))
            T4 = smith.zeros((n, n))
            T5 = smith.zeros((n, n))
            T6 = smith.zeros((n, n))
            T7 = smith.zeros((n, n))
            T8 = smith.zeros((n, n))
            T9 = smith.zeros((n, n))
            T10 = smith.zeros((n, n))
            smith.relu_(T1)
            result = {}
            result["T1"] = smith.mm(A, T1)
            result["T2"] = smith.mm(A, T2)
            result["T3"] = smith.mm(A, T3)
            result["T4"] = smith.mm(A, T4)
            result["T5"] = smith.mm(A, T5)
            result["T6"] = smith.mm(A, T6)
            result["T7"] = smith.mm(A, T7)
            result["T8"] = smith.mm(A, T8)
            result["T9"] = smith.mm(A, T9)
            result["T10"] = smith.mm(A, T10)
            return result

        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)

        FileCheck().check_count("aten::mm", 1, exactly=True).run(test_batch_mm.graph)
        FileCheck().check_count("prim::MMBatchSide", 1, exactly=True).run(
            test_batch_mm.graph
        )

    def test_batch_mm_side_prohibited_mutation_common_side(self):
        @smith.jit.script
        def test_batch_mm(n: int):
            A = smith.zeros((n, n))
            T1 = smith.zeros((n, n))
            T2 = smith.zeros((n, n))
            T3 = smith.zeros((n, n))
            T4 = smith.zeros((n, n))
            T5 = smith.zeros((n, n))
            T6 = smith.zeros((n, n))
            T7 = smith.zeros((n, n))
            T8 = smith.zeros((n, n))
            T9 = smith.zeros((n, n))
            T10 = smith.zeros((n, n))
            smith.relu_(A)
            result = {}
            result["T1"] = smith.mm(A, T1)
            result["T2"] = smith.mm(A, T2)
            result["T3"] = smith.mm(A, T3)
            result["T4"] = smith.mm(A, T4)
            result["T5"] = smith.mm(A, T5)
            result["T6"] = smith.mm(A, T6)
            result["T7"] = smith.mm(A, T7)
            result["T8"] = smith.mm(A, T8)
            result["T9"] = smith.mm(A, T9)
            result["T10"] = smith.mm(A, T10)
            return result

        FileCheck().check_count("aten::mm", 10, exactly=True).run(test_batch_mm.graph)
        self.run_pass("batch_mm", test_batch_mm.graph)
        FileCheck().check_count("aten::mm", 10, exactly=True).check_not(
            "prim::MMBatchSide"
        ).run(test_batch_mm.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
