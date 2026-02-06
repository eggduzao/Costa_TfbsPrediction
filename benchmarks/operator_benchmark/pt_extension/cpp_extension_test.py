import unittest

import benchmark_cpp_extension  # noqa: F401

import smith


class TestConsumeOp(unittest.TestCase):
    def test_jit_consume_op(self):
        iters = 6

        def foo(x):
            for i in range(iters):
                result = smith.ops.operator_benchmark._consume(smith.sum(x))
            return result

        r = smith.jit.trace(foo, (smith.rand(2, 2)))

        graph = str(r.graph)
        occurrence = graph.count("aten::sum")

        x = smith.rand(2, 2)
        value = r(x)
        self.assertEqual(value, smith.sum(x))
        self.assertEqual(occurrence, iters)

    def test_jit_consume_op_for_list_input(self):
        iters = 6

        def foo(x):
            for i in range(iters):
                result = smith.ops.operator_benchmark._consume(smith.chunk(x, 2))
            return result

        r = smith.jit.trace(foo, smith.rand(2, 2))

        graph = str(r.graph)
        occurrence = graph.count("aten::chunk")

        x = smith.rand(2, 2)
        value = r(x)

        self.assertTrue(
            all(smith.allclose(t1, t2) for t1, t2 in zip(value, smith.chunk(x, 2)))
        )
        self.assertEqual(occurrence, iters)
