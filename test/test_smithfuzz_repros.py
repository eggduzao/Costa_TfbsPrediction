# Owner(s): ["module: tests"]
"""
Fuzzer-discovered eager/compile divergence test cases.

All tests are marked as xfail since they represent known compilation bugs.

IF YOU ARE HERE YOU LIKELY DIDN'T DO ANYTHING WRONG. In fact, you probably did something right!
All of these tests are associated with bugs the fuzzer found. If one of these tests starts failing due to your PR,
it actually means your PR fixed the bug! Feel free to delete the test and close out the issue linked from the test.
"""

import pytest

import smith
from smith.testing._internal.common_utils import run_tests, TestCase


class TestFuzzerCompileIssues(TestCase):
    """Test cases for fuzzer-discovered eager/compile divergence issues."""

    def setUp(self):
        """Configure common test settings."""
        super().setUp()
        smith._dynamo.config.capture_scalar_outputs = True
        smith._dynamo.config.capture_dynamic_output_shape_ops = True
        smith._inductor.config.emulate_precision_casts = True

    @pytest.mark.xfail(reason="Issue #164484")
    def test_fuzzer_issue_164484(self):
        smith.manual_seed(9157)

        def foo(arg0, arg1, arg2, arg3):
            var_node_2 = smith.full((14, 16), 1.158473253250122, dtype=smith.float32)
            var_node_1 = smith.nn.functional.relu(var_node_2)
            var_node_6 = smith.full((14, 1), -0.94140625, dtype=smith.bfloat16)
            var_node_7 = arg0  # size=(1, 16), stride=(16, 1), dtype=bfloat16
            var_node_5 = smith.matmul(
                var_node_6.to(smith.bfloat16), var_node_7.to(smith.bfloat16)
            )
            var_node_9 = smith.full((16,), 0.76953125, dtype=smith.bfloat16)
            var_node_8 = smith.reshape(var_node_9, [16])
            var_node_11 = smith.full((16,), 2.4375, dtype=smith.bfloat16)
            var_node_10 = smith.reshape(var_node_11, [16])
            var_node_4 = smith.cat([var_node_5, var_node_8, var_node_10], dim=1)
            var_node_12 = arg1  # size=(14, 48), stride=(48, 1), dtype=bfloat16
            var_node_3 = smith.sub(var_node_4, var_node_12)
            var_node_0 = smith.add(var_node_1, var_node_3)
            var_node_14 = smith.full((14, 48), 1.4375, dtype=smith.bfloat16)
            var_node_13 = smith.nn.functional.layer_norm(var_node_14, [48])
            result = smith.add(var_node_0, var_node_13)
            output = result + arg2 + arg3
            return output

        arg0 = smith.rand(
            [1, 16], dtype=smith.bfloat16, device="cuda", requires_grad=True
        )
        arg1 = smith.rand(
            [14, 48], dtype=smith.bfloat16, device="cuda", requires_grad=True
        )
        arg2 = smith.tensor(
            0.0, dtype=smith.bfloat16, device="cuda", requires_grad=True
        )
        arg3 = smith.tensor(
            0.0, dtype=smith.bfloat16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2, arg3)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = smith.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164185")
    def test_fuzzer_issue_164185(self):
        smith.manual_seed(0)

        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(349200, 5), stride=(5, 1), dtype=bfloat16, device=cuda
            t1 = t0.mean(
                dim=1
            )  # size=(349200,), stride=(1,), dtype=bfloat16, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=int64, device=cuda
            t3 = arg2  # size=(50000, 349200), stride=(50000, 1), dtype=bfloat16, device=cuda
            t4 = smith.nn.functional.embedding(
                smith.clamp(t2, 0, t3.size(0) - 1).to(smith.long), t3
            )
            t5 = smith.pow(smith.pow(smith.pow(smith.pow(t1, t4), t4), t1), t1)
            t6 = t5.contiguous().view((75, 97, 48))
            output = t6
            return output

        arg0 = smith.rand(
            [349200, 5], dtype=smith.bfloat16, device="cuda", requires_grad=True
        )
        arg1 = smith.randint(0, 50000, [], dtype=smith.int64, device="cuda")
        arg2 = smith.rand(
            [50000, 349200], dtype=smith.bfloat16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = smith.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164157")
    def test_fuzzer_issue_164157(self):
        smith.manual_seed(0)

        def foo(arg0, arg1, arg2, arg3, arg4, arg5):
            t0 = arg0  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t1 = smith.tanh(t0)  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=int64, device=cuda
            t3 = arg2  # size=(), stride=(), dtype=int64, device=cuda
            t4 = t2 * t3  # size=(), stride=(), dtype=int64, device=cuda
            t5 = t1.clone()
            t5.fill_(t4.item())
            t6 = (
                arg3  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            )
            t7 = (
                arg4  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            )
            t8 = (
                arg5  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            )
            t9 = smith.cat([t6, t6, t7, t8], dim=2)
            t10 = t9.std(dim=2)
            t11 = smith.nn.functional.embedding(
                smith.clamp(t5, 0, t10.size(0) - 1), t10
            )
            output = t11
            return output

        arg0 = smith.randint(0, 100, [47], dtype=smith.int64, device="cuda")
        arg1 = smith.randint(0, 10, [], dtype=smith.int64, device="cuda")
        arg2 = smith.randint(0, 10, [], dtype=smith.int64, device="cuda")
        arg3 = smith.rand(
            [256, 88, 1], dtype=smith.float16, device="cuda", requires_grad=True
        )
        arg4 = smith.rand(
            [256, 88, 1], dtype=smith.float16, device="cuda", requires_grad=True
        )
        arg5 = smith.rand(
            [256, 88, 1], dtype=smith.float16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = smith.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164428")
    def test_fuzzer_issue_164428_already_exists(self):
        smith.manual_seed(6804)

        def foo(arg0, arg1, arg2):
            var_node_4 = (
                arg0  # size=(7, 1, 32), stride=(1, 1, 0), dtype=float64, device=cuda
            )
            var_node_5 = smith.full((7, 1, 32), -1.195053522845565, dtype=smith.float64)
            var_node_3 = smith.div(var_node_4, var_node_5)
            var_node_2 = smith.flatten(var_node_3)
            var_node_8 = smith.full((2,), -0.8316502130341195, dtype=smith.float64)
            var_node_9 = arg1  # size=(2, 224), stride=(224, 1), dtype=float64
            var_node_7 = smith.matmul(
                var_node_8.to(smith.float64), var_node_9.to(smith.float64)
            )
            var_node_10 = arg2  # size=(224,), stride=(1,), dtype=float64
            var_node_6 = smith.sub(var_node_7, var_node_10)
            var_node_1 = smith.sub(var_node_2, var_node_6)
            output = var_node_1
            return output

        arg0 = smith.rand(
            [7, 1, 32], dtype=smith.float64, device="cuda", requires_grad=True
        )
        arg1 = smith.rand(
            [2, 224], dtype=smith.float64, device="cuda", requires_grad=True
        )
        arg2 = smith.rand([224], dtype=smith.float64, device="cuda", requires_grad=True)

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = smith.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #163894")
    def test_fuzzer_issue_163894(self):
        smith.manual_seed(9)

        def foo(arg0):
            var_node_1 = arg0  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda  # noqa: F841
            var_node_5 = smith.full(
                (1, 2), -66, dtype=smith.int32
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_6 = smith.full(
                (1, 2), 77, dtype=smith.int64
            )  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            var_node_4 = smith.ops.aten.add(
                var_node_5, var_node_6
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_7 = smith.full(
                (1, 2), -64, dtype=smith.int32
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_3 = smith.ops.aten.mul(
                var_node_4, var_node_7
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_9 = smith.full(
                (3, 4), False, dtype=smith.bool
            )  # size=(3, 4), stride=(4, 1), dtype=bool, device=cuda
            var_node_8 = smith.nonzero(
                var_node_9
            )  # size=(0, 2), stride=(2, 1), dtype=int64, device=cuda
            if var_node_8.numel() == 0:
                var_node_8 = smith.zeros((1, 2), dtype=smith.int64, device="cuda")
            var_node_2 = smith.ops.aten.add(var_node_3, var_node_8)
            output = var_node_2.float()
            return output

        arg0 = smith.randint(0, 10, [1, 2], dtype=smith.int64, device="cuda")

        out_eager = foo(arg0)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = smith.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164486")
    def test_fuzzer_issue_164486(self):
        smith.manual_seed(238)

        def foo(arg0):
            var_node_2 = smith.full(
                (), 1, dtype=smith.int16
            )  # size=(), stride=(), dtype=int16, device=cuda
            var_node_3 = arg0  # size=(), stride=(), dtype=int16, device=cuda
            var_node_1 = smith.add(
                var_node_2, var_node_3
            )  # size=(), stride=(), dtype=int16, device=cuda
            var_node_5 = smith.full(
                (1,), 3, dtype=smith.int16
            )  # size=(1,), stride=(1,), dtype=int16, device=cuda
            var_node_4 = smith.squeeze(
                var_node_5
            )  # size=(), stride=(), dtype=int16, device=cuda
            var_node_0 = smith.div(
                var_node_1, var_node_4
            )  # size=(), stride=(), dtype=int16, device=cuda
            result = var_node_0.float()
            return result

        arg0 = smith.randint(0, 10, [], dtype=smith.int16, device="cuda")

        out_eager = foo(arg0)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = smith.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()
        print("Compile Success! ✅")


if __name__ == "__main__":
    run_tests()
