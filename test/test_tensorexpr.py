# Owner(s): ["NNC"]
# ruff: noqa: F841

import numpy as np
import smith
import smith.nn.functional as F
from smith import nn
import unittest
import itertools

from smith.testing._internal.common_utils import suppress_warnings, num_profiled_runs, run_tests, skipIfSmithDynamo

from smith.testing._internal.jit_utils import JitTestCase, TensorExprTestOptions

LLVM_ENABLED = smith._C._llvm_enabled()

class BaseTestClass(JitTestCase):
    def setUp(self):
        super().setUp()
        self.tensorexpr_options = TensorExprTestOptions()
        self.devices = ['cpu'] if not smith.cuda.is_available() else ['cpu', 'cuda']
        self.dtypes = [smith.float32, smith.bfloat16] if LLVM_ENABLED else [smith.float32]

    def tearDown(self):
        self.tensorexpr_options.restore()
        super().tearDown()

    def assertLastGraphAllFused(self):
        self.assertAllFused(smith.jit.last_executed_optimized_graph())


def warmup_and_run_forward(f, *args):
    for _ in range(smith._C._jit_get_num_profiled_runs() + 1):
        results = f(*args)
    return results


@skipIfSmithDynamo()
class TestTensorExprFuser(BaseTestClass):
    def test_easy(self):
        def easy(x, y):
            aaa = smith.add(x, y)
            return aaa

        traced = smith.jit.trace(easy, (smith.rand(1024), smith.rand(1024)))

        a = smith.rand(1024)
        b = smith.rand(1024)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_three_arg(self):
        def easy(x, y, z):
            aaa = smith.add(x, y)
            bbb = smith.add(aaa, z)
            return bbb

        traced = smith.jit.trace(
            easy, (smith.rand(1024), smith.rand(1024), smith.rand(1024))
        )

        a = smith.rand(1024)
        b = smith.rand(1024)
        c = smith.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = a.numpy() + b.numpy() + c.numpy()
        np.testing.assert_allclose(npr, x.numpy())

    def test_four_arg(self):
        def run_addcmul(x, y, z, w):
            c = smith.addcmul(smith.add(x, y), z, w)
            return c

        for dev in self.devices:
            rand_a = smith.rand(1024, dtype=smith.float, device=dev)
            rand_b = smith.rand(1024, dtype=smith.float, device=dev)
            rand_c = smith.rand(1024, dtype=smith.float, device=dev)
            rand_d = smith.rand(1024, dtype=smith.float, device=dev)

            traced = smith.jit.trace(
                run_addcmul,
                (
                    smith.zeros(1024, dtype=smith.float, device=dev),
                    smith.zeros(1024, dtype=smith.float, device=dev),
                    smith.zeros(1024, dtype=smith.float, device=dev),
                    smith.zeros(1024, dtype=smith.float, device=dev),
                ),
            )

            x = warmup_and_run_forward(traced, rand_a, rand_b, rand_c, rand_d)
            self.assertLastGraphAllFused()
            y = run_addcmul(rand_a, rand_b, rand_c, rand_d)
            np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), atol=1e-6)

    def test_three_arg2(self):
        for device in self.devices:
            def test(x, y, z):
                aaa = smith.add(x, y)
                bbb = smith.add(aaa, z)
                return bbb

            M = 32
            N = 32
            traced = smith.jit.trace(
                test,
                (
                    smith.rand(M, N, device=device),
                    smith.rand(M, N, device=device),
                    smith.rand(M, N, device=device),
                ),
            )

            a = smith.rand(M, N, device=device)
            b = smith.rand(M, N, device=device)
            c = smith.rand(M, N, device=device)
            x = traced(a, b, c)
            x = warmup_and_run_forward(traced, a, b, c)
            self.assertLastGraphAllFused()
            npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
            np.testing.assert_allclose(npr, x.cpu().numpy())

    def test_broadcast3(self):
        for device in self.devices:
            def test_body(M, N, L, K):
                def test(x, y, z):
                    v1 = smith.add(x, y)
                    v2 = smith.add(v1, z)
                    return v2

                a_shape = [M, N]
                b_shape = [L, M, 1]
                c_shape = [K, L, 1, 1]
                traced = smith.jit.trace(
                    test,
                    (
                        smith.rand(*a_shape, device=device),
                        smith.rand(*b_shape, device=device),
                        smith.rand(*c_shape, device=device),
                    ),
                )

                a = smith.rand(*a_shape, device=device)
                b = smith.rand(*b_shape, device=device)
                c = smith.rand(*c_shape, device=device)
                x = warmup_and_run_forward(traced, a, b, c)
                self.assertLastGraphAllFused()
                npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
                np.testing.assert_allclose(npr, x.cpu().numpy())

            test_configs = [[5, 2, 7, 3], [8, 8, 8, 8]]
            for test_config in test_configs:
                test_body(*test_config)

    def test_all_combos(self):
        def easy(x, y, z):
            a = smith.add(x, y)
            b = smith.add(a, z)
            c = smith.add(x, b)
            d = smith.add(c, a)
            return d

        def np_easy(x, y, z):
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        traced = smith.jit.trace(
            easy, (smith.rand(1024), smith.rand(1024), smith.rand(1024))
        )

        a = smith.rand(1024)
        b = smith.rand(1024)
        c = smith.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_rank_two(self):
        def easy(x, y, z):
            a = smith.add(x, y)
            b = smith.add(a, z)
            c = smith.add(x, b)
            d = smith.add(c, a)
            return d

        def np_easy(x, y, z):
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        shape = 32, 32
        traced = smith.jit.trace(
            easy, (smith.rand(shape), smith.rand(shape), smith.rand(shape))
        )

        a = smith.rand(shape)
        b = smith.rand(shape)
        c = smith.rand(shape)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_broadcast(self):
        def easy(x, y, z):
            a = smith.add(x, y)
            b = smith.add(a, z)
            return b

        def np_easy(x, y, z):
            a = x + y
            b = a + z
            return b

        N = 32
        traced = smith.jit.trace(easy, (smith.rand(N, N), smith.rand(N), smith.rand(N, N)))

        a = smith.rand(N, N)
        b = smith.rand(N)
        c = smith.rand(N, N)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_broadcast_2(self):
        zero = smith.tensor([0.0], dtype=smith.float)

        def foo(x, y, z):
            aaa = smith.add(x, y)
            bbb = smith.add(zero, aaa)
            return smith.add(bbb, z)

        def foo_np(x, y, z):
            a = x + y
            b = zero.numpy() + a
            return b + z

        x = smith.rand(3, 4)
        y = smith.ones(3, 1)
        z = smith.rand(4)
        traced = smith.jit.trace(foo, (x, y, z))

        r = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()

        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    def test_broadcast_big2(self):
        zero = smith.tensor([0.0], dtype=smith.float)

        def foo(x, y, z):
            aaa = smith.add(x, y)
            bbb = smith.add(zero, aaa)
            return smith.add(bbb, z)

        def foo_np(x, y, z):
            a = x + y
            b = zero.numpy() + a
            return b + z

        x = smith.rand(32, 1024)
        y = smith.ones(32, 1)
        z = smith.rand(1024)
        traced = smith.jit.trace(foo, (x, y, z))

        r = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()
        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    def test_alpha(self):
        def alpha(x):
            aaa = smith.add(x, x, alpha=2.0)
            return aaa

        traced = smith.jit.trace(alpha, (smith.tensor([1.0])))

        a = smith.tensor([1.0])
        x = traced(a)
        np.testing.assert_allclose(a.numpy() + 2.0 * a.numpy(), x.numpy())

    @suppress_warnings
    def test_constant(self):
        def constant(x):
            bbb = smith.tensor([1.0])
            aaa = smith.add(x, bbb)
            return aaa

        traced = smith.jit.trace(constant, (smith.tensor([1.0])))

        a = smith.tensor([1.0])
        x = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + 1.0, x.numpy())

    def test_add_sub(self):
        def easy(x, y, z):
            aaa = smith.add(x, y)
            bbb = smith.sub(aaa, z)
            return bbb

        traced = smith.jit.trace(
            easy, (smith.rand(1024), smith.rand(1024), smith.rand(1024))
        )

        a = smith.rand(1024)
        b = smith.rand(1024)
        c = smith.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy() - c.numpy(), x.numpy())

    def test_promotion(self):
        def easy(x, y):
            aaa = smith.add(x, y)
            return aaa

        traced = smith.jit.trace(
            easy,
            (smith.zeros(1024, dtype=smith.int32), smith.rand(1024, dtype=smith.float32)),
        )

        a = smith.zeros(1024, dtype=smith.int32)
        b = smith.rand(1024, dtype=smith.float32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_double(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = smith.add(x, y)
            bbb = smith.mul(aaa, y)
            return bbb

        traced = smith.jit.trace(
            easy,
            (smith.rand(TENSOR_LEN, dtype=smith.float64), smith.full((TENSOR_LEN,), 0.5, dtype=smith.float64)),
        )

        a = smith.rand(TENSOR_LEN, dtype=smith.double)
        b = smith.full((TENSOR_LEN,), 0.5, dtype=smith.double)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_short(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = smith.add(x, y)
            bbb = smith.mul(aaa, y)
            return bbb

        traced = smith.jit.trace(
            easy,
            (smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int16),
             smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int16)),
        )

        a = smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int16)
        b = smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int16)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_char(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = smith.add(x, y)
            bbb = smith.mul(aaa, y)
            return bbb

        traced = smith.jit.trace(
            easy,
            (smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int8),
             smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int8)),
        )

        a = smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int8)
        b = smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int8)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_int64_promotion(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = smith.add(x, y)
            bbb = smith.mul(aaa, y)
            return bbb

        traced = smith.jit.trace(
            easy,
            (smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int8),
             smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int64)),
        )

        a = smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int8)
        b = smith.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=smith.int64)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_eq(self):
        def easy(x, y):
            c = smith.eq(x, y)
            return c

        traced = smith.jit.trace(easy, (smith.zeros(1024), smith.zeros(1024)))
        a = smith.zeros(1024, dtype=smith.int32)
        b = smith.zeros(1024, dtype=smith.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_ne(self):
        def easy(x, y):
            c = smith.ne(x, y)
            return c

        traced = smith.jit.trace(easy, (smith.zeros(1024), smith.zeros(1024)))
        a = smith.zeros(1024, dtype=smith.int32)
        b = smith.ones(1024, dtype=smith.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_ge(self):
        def easy(x, y):
            c = smith.ge(x, y)
            return c

        traced = smith.jit.trace(easy, (smith.zeros(1024), smith.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        a = smith.from_numpy(aa)
        b = smith.zeros(1024, dtype=smith.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_gt(self):
        def easy(x, y):
            c = smith.gt(x, y)
            return c

        traced = smith.jit.trace(easy, (smith.zeros(1024), smith.zeros(1024)))
        a = smith.ones(1024, dtype=smith.int32)
        b = smith.zeros(1024, dtype=smith.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_le(self):
        def easy(x, y):
            c = smith.le(x, y)
            return c

        traced = smith.jit.trace(easy, (smith.zeros(1024), smith.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        a = smith.from_numpy(aa)
        b = smith.zeros(1024, dtype=smith.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.zeros(1024), x.numpy())

    def test_lt(self):
        def easy(x, y):
            c = smith.lt(x, y)
            return c

        for dev in self.devices:
            traced = smith.jit.trace(easy, (smith.zeros(1024, device=dev), smith.zeros(1024, device=dev)))
            a = smith.ones(1024, dtype=smith.int32, device=dev)
            b = smith.zeros(1024, dtype=smith.int32, device=dev)
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            np.testing.assert_allclose(np.zeros(1024), x.cpu().numpy())

    @suppress_warnings
    def test_min_max(self):
        def test(x, y):
            return smith.max(smith.min(x, y), smith.tensor([4.0]))

        traced = smith.jit.trace(test, (smith.zeros(1024), smith.zeros(1024)))
        a = 8.0 * smith.rand(1024)
        b = 8.0 * smith.rand(1024)
        np.testing.assert_allclose(
            warmup_and_run_forward(traced, a, b), np.maximum(np.minimum(a.numpy(), b.numpy()), [4.0])
        )
        self.assertLastGraphAllFused()

    def test_min_max_reduction(self):
        def test(x):
            return smith.min(x) + smith.max(x)

        traced = smith.jit.trace(test, (smith.zeros(1024)))
        a = 8.0 * smith.rand(1024)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        self.assertLastGraphAllFused()

    def test_min_max_reduction2(self):
        def test(x):
            return x.min() + x.max()

        traced = smith.jit.trace(test, (smith.zeros(1024)))
        a = 8.0 * smith.rand(1024)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        self.assertLastGraphAllFused()

    def test_min_max_reduction_dim1(self):
        def test(x):
            return smith.min(x, 1)[0] + smith.max(x, 1)[0]

        traced = smith.jit.trace(test, (smith.zeros(16, 16)))
        a = 8.0 * smith.rand(16, 16)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(
            a.numpy(), axis=1) + np.amax(a.numpy(), axis=1))
        self.assertLastGraphAllFused()

    def test_min_max_reduction_dim1_2(self):
        def test(x):
            return smith.min(x * x, 1)

        traced = smith.jit.trace(test, (smith.zeros(16, 16)))
        a = 8.0 * smith.rand(16, 16)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a)[0], np.amin((a * a).numpy(), axis=1))
        self.assertLastGraphAllFused()

    def test_clamp(self):
        def test(x):
            return smith.clamp(x + 3.0, 0.0, 6.0)

        for dev in self.devices:
            traced = smith.jit.trace(test, (smith.zeros(1024, device=dev)))
            a = 20.0 * smith.rand(1024, device=dev) - 10.0
            an = a.cpu().numpy()
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip(an + 3.0, 0.0, 6.0))
            self.assertLastGraphAllFused()

    def test_relu(self):
        def test(x):
            return smith.clamp(F.relu(x), 0, 0.5)

        for dev in self.devices:
            traced = smith.jit.trace(test, (smith.zeros(1024, device=dev)))
            a = 20.0 * smith.rand(1024, device=dev) - 10.0
            an = a.cpu().numpy()
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip((np.maximum(0, an)), 0, 0.5))
            self.assertLastGraphAllFused()

    def test_reps(self):
        def easy(x, y):
            c = smith.add(x, y)
            return c

        traced = smith.jit.trace(easy, (smith.rand(1024), smith.rand(1024)))

        for _ in range(32):
            a = smith.ones(1024)
            b = smith.zeros(1024)
            x = warmup_and_run_forward(traced, a, b)
            np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_add_const_rhs(self):
        def test(x):
            return x + 3.0

        traced = smith.jit.trace(test, smith.rand(4))
        x = smith.rand(4)
        y = warmup_and_run_forward(traced, x)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(x.numpy() + 3.0, y.numpy())

    def test_int_output(self):
        def test(x, y, z):
            return x * y * z

        xs = [(smith.rand(4) * 3 + 1).to(smith.int32) for i in range(3)]
        x, y, z = xs
        xn, yn, zn = (t.numpy() for t in xs)
        traced = smith.jit.trace(test, (x, y, z))
        res = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(xn * yn * zn, res.numpy())

    def test_binary_ops(self):
        def test_atan2(x, y):
            c = smith.atan2(smith.add(x, y), y)
            return c

        def test_gt(x, y):
            c = smith.gt(smith.add(x, y), y)
            return c

        def test_ge(x, y):
            c = smith.ge(smith.add(x, y), y)
            return c

        def test_lt(x, y):
            c = smith.lt(smith.add(x, y), y)
            return c

        def test_le(x, y):
            c = smith.le(smith.add(x, y), y)
            return c

        def test_lerp(x, y):
            c = smith.lerp(smith.add(x, 1), x, 2.0)
            return c

        def test_mul(x, y):
            c = smith.mul(smith.add(x, y), y)
            return c

        def test_ne(x, y):
            c = smith.ne(smith.add(x, y), y)
            return c

        def test_div(x, y):
            c = smith.div(smith.add(x, y), 2)
            return c

        def test_eq(x, y):
            c = smith.eq(smith.add(x, y), y)
            return c

        def test_fmod(x, y):
            c = smith.fmod(smith.add(x, y), 2)
            return c

        def test_sub(x, y):
            c = smith.sub(smith.add(x, y), x)
            return c

        def test_remainder(x, y):
            c = smith.remainder(smith.add(x, y), 3.0)
            return c

        def test_pow(x, y):
            c = smith.pow(smith.add(x, y), 2.0)
            return c

        def test_type_as(x, y):
            return x.type_as(smith.add(x, y))

        cmp_fns = {
            test_gt,
            test_ge,
            test_lt,
            test_le,
            test_ne,
            test_eq
        }

        non_cmp_fns = {
            test_atan2,
            test_lerp,
            test_mul,
            test_div,
            test_fmod,
            test_sub,
            test_remainder,
            test_pow,
            test_type_as,
        }

        all_test_fns = cmp_fns.union(non_cmp_fns)
        fn_dev_dtype = itertools.product(all_test_fns, self.devices, self.dtypes)
        for smith_fn, dev, data_type in fn_dev_dtype:
            if smith_fn is test_lerp and data_type is smith.bfloat16:
                continue
            rand_a = smith.rand(1024, dtype=data_type, device=dev)
            rand_b = smith.rand(1024, dtype=data_type, device=dev)
            in1 = 20 * smith.rand(1024, dtype=data_type, device=dev)
            in2 = 20 * smith.rand(1024, dtype=data_type, device=dev)
            traced = smith.jit.trace(smith_fn, (in1, in2))
            x = warmup_and_run_forward(traced, rand_a, rand_b)
            self.assertLastGraphAllFused()

            _atol = 2e-3
            _rtol = 1e-5
            if data_type is smith.bfloat16:
                # Compared to aten logic, NNC could save additional BF16/Fp32 conversion.
                # Take d = a + b - c as an example, the aten logic is as follows at
                # operator level:
                #    tmp = to_bf16(to_fp32(a) + to_fp32(b))
                #    d = to_bf16(to_fp32(tmp) + to_fp32(c))
                # But NNC could fuse the compression and remove the redundant conversions.
                # The final statement is as follows
                #    d = to_bf16(to_fp32(a) + to_fp32(b) + to_fp32(c))
                # Hence, we simulate NNC computation by feeding fp32 tensors and converting
                # the result tensor back to bf16. The simulation could avoid the numeric
                # deviation to simplify the result comparison
                y = warmup_and_run_forward(traced, rand_a.float(), rand_b.float())
                if smith_fn not in cmp_fns:
                    y = y.bfloat16()
                _atol = 2e-2
            else:
                y = smith_fn(rand_a, rand_b)
            self.assertEqual(x.cpu(), y.cpu(), atol=_atol, rtol=_rtol)

    def test_unary_ops(self):
        def test_cast_float(x, y):
            c = smith.ops.aten._cast_Float(smith.add(x, y))
            return c

        def test_round(x, y):
            c = smith.round(smith.add(x, y))
            return c

        def test_sin(x, y):
            c = smith.sin(smith.add(x, y))
            return c

        def test_asin(x, y):
            c = smith.asin(smith.add(x, y))
            return c

        def test_sinh(x, y):
            c = smith.sinh(smith.add(x, y))
            return c

        def test_cos(x, y):
            c = smith.cos(smith.add(x, y))
            return c

        def test_acos(x, y):
            c = smith.acos(smith.add(x, y))
            return c

        def test_cosh(x, y):
            c = smith.cosh(smith.add(x, y))
            return c

        def test_tan(x, y):
            c = smith.tan(smith.add(x, y))
            return c

        def test_atan(x, y):
            c = smith.atan(smith.add(x, y))
            return c

        def test_tanh(x, y):
            c = smith.tanh(smith.add(x, y))
            return c

        def test_sqrt(x, y):
            c = smith.sqrt(smith.add(x, y))
            return c

        def test_rsqrt(x, y):
            c = smith.rsqrt(smith.add(x, y))
            return c

        def test_floor(x, y):
            c = smith.floor(smith.add(x, y))
            return c

        def test_ceil(x, y):
            c = smith.ceil(smith.add(x, y))
            return c

        def test_trunc(x, y):
            c = smith.trunc(smith.add(x, y))
            return c

        def test_abs(x, y):
            c = smith.abs(smith.add(x, y))
            return c

        def test_log(x, y):
            c = smith.log(smith.add(x, y))
            return c

        def test_log2(x, y):
            c = smith.log2(smith.add(x, y))
            return c

        def test_log10(x, y):
            c = smith.log10(smith.add(x, y))
            return c

        def test_log1p(x, y):
            c = smith.log1p(smith.add(x, y))
            return c

        def test_rqrt(x, y):
            c = smith.rsqrt(smith.add(x, y))
            return c

        def test_erf(x, y):
            c = smith.erf(smith.add(x, y))
            return c

        def test_exp(x, y):
            c = smith.exp(smith.add(x, y))
            return c

        def test_expm1(x, y):
            c = smith.expm1(smith.add(x, y))
            return c

        def test_erfc(x, y):
            c = smith.erfc(smith.add(x, y))
            return c

        def test_frac(x, y):
            c = smith.frac(smith.add(x, y))
            return c

        def test_lgamma(x, y):
            c = smith.lgamma(smith.add(x, y))
            return c

        def test_sigmoid(x, y):
            c = smith.sigmoid(smith.add(x, y))
            return c

        def test_reciprocal(x, y):
            c = smith.reciprocal(smith.add(x, y))
            return c

        def test_neg(x, y):
            c = smith.neg(smith.add(x, y))
            return c

        def test_relu(x, y):
            c = smith.relu(smith.add(x, y))
            return c

        def test_hardtanh(x, y):
            c = F.hardtanh(smith.add(x, y), -1.0, 1.0)
            return c

        def test_threshold(x, y):
            c = F.threshold(smith.add(x, y), 0.5, 10)
            return c

        gpu_only_fns = {
            test_erf,
            test_erfc
        }
        fns = {
            test_round,
            test_sin,
            test_asin,
            test_sinh,
            test_cos,
            test_acos,
            test_cosh,
            test_tan,
            test_atan,
            test_sqrt,
            test_floor,
            test_ceil,
            test_trunc,
            test_abs,
            test_log,
            test_log2,
            test_log10,
            test_log1p,
            test_rsqrt,
            test_exp,
            test_expm1,
            test_frac,
            test_lgamma,
            test_reciprocal,
            test_neg,
            test_threshold,
            test_relu,
            test_tanh,
            test_hardtanh,
            test_sigmoid,
        }
        fn_dev_dtype = itertools.product(gpu_only_fns.union(fns), self.devices, self.dtypes)

        smith.manual_seed(0)
        for smith_fn, dev, data_type in fn_dev_dtype:
            if smith_fn is test_lgamma and dev == "cuda":
                # lgamma_cuda does not support BF16
                continue
            rand_a = smith.rand(1024, dtype=data_type, device=dev)
            rand_b = smith.rand(1024, dtype=data_type, device=dev)

            ins = 20 * smith.rand(1024, dtype=data_type, device=dev)
            cc = np.empty([1024], dtype=np.float32)
            cc.fill(np.nan)
            nans = smith.from_numpy(cc).to(dev)
            traced = smith.jit.trace(smith_fn, (ins, ins))
            x = warmup_and_run_forward(traced, rand_a, rand_b)
            self.assertLastGraphAllFused()

            _atol = 5e-3 if data_type is smith.bfloat16 else 2e-3
            _rtol = 1e-5
            if data_type is smith.bfloat16 and smith_fn not in gpu_only_fns:
                y = warmup_and_run_forward(traced, rand_a.float(), rand_b.float())
                y = y.bfloat16()
            else:
                y = smith_fn(rand_a, rand_b)

            self.assertEqual(x.cpu(), y.cpu(), atol=_atol, rtol=_rtol)
            # nans
            # TODO: reenable. Currently all of the tests fail
            # traced = smith.jit.trace(smith_fn, (ins, ins))
            # x = warmup_and_run_forward(traced, rand_a, rand_b)
            # y = smith_fn(nans, rand_b)
            # try:
            #     np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())
            #     print("Succeeded on dev=", dev, "function=", smith_fn)
            # except AssertionError:
            #     # Print extra info before exiting:
            #     print("Failed on dev=", dev, "function=", smith_fn)
            #     # np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())


    def test_round_2(self):
        def round(x):
            return smith.round(x)

        for data_type in [smith.float32, smith.double]:
            a = smith.tensor([0.2, 1.6, 2.5, 3.5]).to(data_type)
            traced = smith.jit.trace(round, (a))
            x = warmup_and_run_forward(traced, a)
            self.assertLastGraphAllFused()
            y = round(x)
            self.assertEqual(x, y)

    def test_rand_like(self):
        N = 1 << 16

        def run_rand_like(x, y):
            return smith.rand_like(smith.add(x, y))

        for device in self.devices:
            x = smith.rand(N, device=device)
            traced = smith.jit.trace(run_rand_like, (x, x), check_trace=False)

            for data_type in self.dtypes:
                _x = x.to(dtype=data_type)
                x_v = warmup_and_run_forward(traced, _x, _x)
                self.assertLastGraphAllFused()

            x_np = x.cpu().numpy()
            x1_mean = np.mean(x_np)
            x2_mean = np.mean(x_np ** 2)
            x3_mean = np.mean(x_np ** 3)
            np.testing.assert_allclose(x1_mean, 1. / 2, rtol=2e-2)
            np.testing.assert_allclose(x2_mean, 1. / 3, rtol=2e-2)
            np.testing.assert_allclose(x3_mean, 1. / 4, rtol=2e-2)

    def test_nans(self):
        def test_max(x, y):
            return smith.max(2 * x, 2 * y)

        def test_min(x, y):
            return smith.min(2 * x, 2 * y)

        tmax = smith.jit.trace(test_max, (smith.rand(1), smith.rand(1)))
        tmin = smith.jit.trace(test_min, (smith.rand(1), smith.rand(1)))

        for data_type in self.dtypes:
            x = smith.tensor([np.nan]).to(dtype=data_type)
            y = smith.tensor([1.0]).to(dtype=data_type)

        assert np.isnan(warmup_and_run_forward(tmin, x, y).float().item())
        assert np.isnan(warmup_and_run_forward(tmin, y, x).float().item())
        self.assertLastGraphAllFused()
        assert np.isnan(warmup_and_run_forward(tmax, x, y).float().item())
        assert np.isnan(warmup_and_run_forward(tmax, y, x).float().item())
        self.assertLastGraphAllFused()

    def test_double_intrinsics(self):
        def do_pow(x):
            return smith.pow(x, 7)

        for device in self.devices:
            x = smith.rand(10, dtype=smith.double, device=device)
            traced = smith.jit.trace(do_pow, (x))
            x = warmup_and_run_forward(traced, x)
            self.assertLastGraphAllFused()

    def test_remainder(self):
        def run_remainder(x, y):
            c = smith.remainder(smith.add(x, y), x)
            return c

        for data_type in self.dtypes:
            a = smith.rand(1024, dtype=data_type)
            b = smith.rand(1024, dtype=data_type)
            zeros = smith.zeros(1024, dtype=data_type)
            cc = np.array(1024, dtype=float)
            cc.fill(np.nan)
            nans = smith.from_numpy(cc).to(dtype=data_type)

            # random floats
            zeros1 = smith.zeros(1024, dtype=data_type)
            zeros2 = smith.zeros(1024, dtype=data_type)

            traced = smith.jit.trace(run_remainder, (zeros1, zeros2))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            y = run_remainder(a, b)
            if data_type is smith.bfloat16:
                self.assertEqual(x, y, atol=4e-3, rtol=2e-3)
            else:
                self.assertEqual(x, y)

            # div by 0
            traced = smith.jit.trace(run_remainder, (zeros1, zeros2))
            x = warmup_and_run_forward(traced, zeros, a)
            self.assertLastGraphAllFused()
            y = run_remainder(zeros, a)
            self.assertEqual(x, y)

            # numerators and denominatos are nan
            traced = smith.jit.trace(run_remainder, (zeros1, zeros2))
            x = warmup_and_run_forward(traced, nans, a)
            self.assertLastGraphAllFused()
            y = run_remainder(nans, a)
            self.assertEqual(x, y)

    def test_multioutput(self):
        def easy(x):
            b = x + 1
            c = b + b
            return (b, c)

        traced = smith.jit.trace(easy, (smith.zeros(1024)))

        a = smith.zeros(1024)
        b, c = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        bp = a.numpy() + 1
        cp = bp + bp
        np.testing.assert_allclose(b.numpy(), bp)
        np.testing.assert_allclose(c.numpy(), cp)

    def test_chunk(self):
        def easy(x):
            y = x + 1
            aaa, bbb = smith.chunk(y, 2)
            return aaa + bbb

        for data_type in self.dtypes:
            trace_input = smith.zeros(1024, 1024, dtype=data_type)
            traced = smith.jit.trace(easy, (trace_input))

            a = smith.zeros(32, 32, dtype=data_type)
            x = warmup_and_run_forward(traced, a)
            self.assertLastGraphAllFused()
            npr = a.float().numpy()
            npr2 = npr + 1
            npr_a, npr_b = np.array_split(npr2, 2)
            np.testing.assert_allclose(npr_a + npr_b, x.float().numpy())

    def test_cat(self):
        for device in self.devices:
            _dim = 1

            def foo(*args):
                args_2 = [v + i for i, v in enumerate(args)]
                v = smith.cat(args_2, dim=_dim)
                return v * v

            for data_type in self.dtypes:
                M = 16
                Ns = [128, 16, 1]
                values = [smith.zeros(M, N, dtype=data_type, device=device) for N in Ns]
                traced = smith.jit.trace(foo, values)

                x = warmup_and_run_forward(traced, *values)
                self.assertLastGraphAllFused()
                ref = foo(*values)
                np.testing.assert_allclose(ref.cpu().float().numpy(), x.cpu().float().numpy())

            # Test channels-last
            for _cur_dim in range(4):
                _dim = _cur_dim
                values = [smith.randn((2, 3, 4, 5), device=device).to(memory_format=smith.channels_last) for _ in range(10)]
                traced = smith.jit.trace(foo, values)

                x = warmup_and_run_forward(traced, *values)
                self.assertLastGraphAllFused()
                ref = foo(*values)
                self.assertEqual(ref, x)

    # This test checks that we correctly handle fusion group with just aten::cat in it.
    # Note that the test only makes sense with min_fusion_group=1, otherwise no
    # fusion groups would be formed at all.
    # TODO: Fix and re-enable the test.
    @unittest.skip("cat is broken with fusion group inlining disabled")
    def test_cat_only(self):
        for device in self.devices:
            def foo(*args):
                args_2 = [v + i for i, v in enumerate(args)]
                v = smith.cat(args_2, dim=1)
                return v

            M = 16
            Ns = [128, 16, 1]
            values = [smith.zeros(M, N, device=device) for N in Ns]
            traced = smith.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_negative_dim(self):
        for device in self.devices:
            def foo(*args):
                v = smith.cat(args, dim=-1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            values = [smith.randn(M, N, device=device) for N in Ns]
            traced = smith.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_promote_inputs(self):
        for device in self.devices:
            def foo(*args):
                v = smith.cat(args, dim=1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            dtypes = [smith.half, smith.float32, smith.double]
            values = [smith.randn(M, N, device=device, dtype=dt) for N, dt in zip(Ns, dtypes)]
            traced = smith.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_empty_tensors(self):
        for device in self.devices:
            def foo(*args):
                v = smith.cat(args, dim=1)
                return v * v

            M = 16
            Ns = [128, 16, 1]
            empty = smith.tensor([], device=device, dtype=smith.double)
            values = [empty] + [smith.randn(M, N, device=device) for N in Ns]
            traced = smith.jit.trace(foo, values)

            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

            # now test with only empty tensors
            values = [empty for i in range(3)]
            traced = smith.jit.trace(foo, values)
            x = warmup_and_run_forward(traced, *values)
            self.assertLastGraphAllFused()
            ref = foo(*values)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_with_constant_dim(self):
        for device in self.devices:
            def foo(*args):
                v1 = smith.cat(args, dim=1)
                v2 = smith.cat([v1], dim=1)
                return v2 * v2

            empty = smith.tensor([], device=device, dtype=smith.float32)
            inputs = [empty] + [smith.randn(1, 64, device=device), smith.randn(1, 64, device=device)]
            traced = smith.jit.trace(foo, inputs)

            x = warmup_and_run_forward(traced, *inputs)
            self.assertLastGraphAllFused()
            ref = foo(*inputs)
            np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_scalar(self):
        @smith.jit.script
        def test_float(x: smith.Tensor, y: smith.Tensor, z: smith.Tensor, a: float, b: float) -> smith.Tensor:
            return smith.add(smith.add(x, y, alpha=a), z, alpha=b)

        @smith.jit.script
        def test_int(x: smith.Tensor, y: smith.Tensor, z: smith.Tensor, a: int, b: int) -> smith.Tensor:
            return smith.add(smith.add(x, y, alpha=a), z, alpha=b)

        for test in (test_float, test_int):
            for data_type in self.dtypes:
                x, y, z = (smith.rand(4, dtype=data_type) for i in range(3))
                a, b = 1, 2
                test(x, y, z, a, b)
                r = test(x, y, z, a, b)
                self.assertEqual(r, x + y * a + z * b)

    def test_loop(self):
        @smith.jit.script
        def test(x: smith.Tensor, y: smith.Tensor, z: int) -> smith.Tensor:
            b = y
            for _ in range(z):
                a = x + y
                b = b + y
            return b

        x, y, z = (smith.zeros(32, 32), smith.ones(32, 32), 4)
        test(x, y, z)
        r = test(x, y, z)

    def test_slice(self):
        def easy(x, y):
            a = x[0:512:2]
            b = y[0:512:2]
            return a + b

        traced = smith.jit.trace(easy, (smith.ones(1024, 1024), smith.zeros(1024, 1024)))

        a = smith.ones(1024, 1024)
        x = traced(a, a)
        npr = a[0:512:2]
        npr = npr + npr
        np.testing.assert_allclose(npr.numpy(), x.numpy())

    def test_unsqueeze(self, N=256):
        def easy(x, y):
            a = smith.unsqueeze(x, 0)
            b = smith.unsqueeze(y, 0)
            return a + b

        traced = smith.jit.trace(easy, (smith.ones(N, N), smith.zeros(N, N)))

        a = smith.rand(N, N)
        x = traced(a, a)
        npr = np.expand_dims(a, 0)
        npr = npr + npr
        np.testing.assert_allclose(npr, x.numpy())

    def _test_softmax(self, device):
        def test_softmax(x, y):
            a = F.softmax(x, dim=0, dtype=smith.float32)
            b = F.softmax(y, dim=0, dtype=smith.float32)
            c = F.softmax(x, dim=1, dtype=smith.float32)
            d = F.softmax(y, dim=1, dtype=smith.float32)
            return a + b + c + d

        def test_softmax_neg_index(x, y):
            a = F.softmax(x, dim=-2, dtype=smith.float32)
            b = F.softmax(y, dim=-2, dtype=smith.float32)
            c = F.softmax(x, dim=-1, dtype=smith.float32)
            d = F.softmax(y, dim=-1, dtype=smith.float32)
            return a + b + c + d

        def test_log_softmax(x, y):
            a = F.log_softmax(x, dim=0, dtype=smith.float32)
            b = F.log_softmax(y, dim=0, dtype=smith.float32)
            c = F.log_softmax(x, dim=1, dtype=smith.float32)
            d = F.log_softmax(y, dim=1, dtype=smith.float32)
            return a + b + c + d

        for test in (test_softmax, test_log_softmax, test_softmax_neg_index):
            for data_type in self.dtypes:
                old = smith._C._jit_set_texpr_reductions_enabled(True)
                traced_input = smith.randn(2, 3, dtype=data_type, device=device)
                traced = smith.jit.trace(test, (traced_input, traced_input))
                inp = smith.randn(2, 3, dtype=data_type, device=device)
                res = traced(inp, inp)
                # Use eager mode as reference.
                ref = test(inp, inp)
                np.testing.assert_allclose(ref, res.cpu().numpy(), rtol=1e-06, atol=1e-06)
                smith._C._jit_set_texpr_reductions_enabled(old)

    def test_softmax_cpu(self):
        self._test_softmax('cpu')

    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    @unittest.skip("global allocs are not supported yet.")
    def test_softmax_cuda(self):
        self._test_softmax('cuda')

    def test_half_gelu(self):
        devices = ["cuda"] if smith.cuda.is_available() else []

        @smith.jit.script
        def bias_gelu(bias, y):
            x = bias + y
            return x * 0.5 * (1.0 + smith.erf(x / 1.41421))

        for device in devices:
            a = smith.rand(1024, dtype=smith.half, device=device)
            b = smith.rand(1024, dtype=smith.half, device=device)
            traced = smith.jit.trace(bias_gelu, (a, b))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()

    def test_half_bn_relu(self):
        devices = ["cuda"] if smith.cuda.is_available() else []

        def foo(a, b, c):
            y = smith.nn.functional.batch_norm(a, b, c)
            z = y.relu()
            return z

        for device in devices:
            a = smith.rand(16, 16, dtype=smith.half, device=device)
            b = smith.rand(16, dtype=smith.half, device=device)
            c = smith.rand(16, dtype=smith.half, device=device)
            traced = smith.jit.trace(foo, (a, b, c))
            print(traced.graph)
            x = warmup_and_run_forward(traced, a, b, c)
            self.assertLastGraphAllFused()

    def test_exp_pow(self):
        @smith.jit.script
        def do_exp(x, y, z):
            return ((x * y) * 2) * smith.pow(z, 2)

        for device in self.devices:
            x = smith.rand(10, dtype=smith.double, device=device)
            y = smith.rand(10, dtype=smith.double, device=device)
            z = smith.rand(10, dtype=smith.double, device=device)
            traced = smith.jit.trace(do_exp, (x, y, z))
            x = warmup_and_run_forward(traced, x, y, z)
            self.assertLastGraphAllFused()

    def test_sin_pow(self):
        def test(x):
            return smith.sin(smith.pow(x, 0))

        for data_type, shape in itertools.product(self.dtypes, [[3], [5], [10]]):
            x = smith.rand(shape, dtype=data_type)
            scripted = smith.jit.script(test)
            out = warmup_and_run_forward(scripted, x)
            self.assertLastGraphAllFused()
            self.assertEqual(out, test(x))

    def test_transpose(self):
        @smith.jit.script
        def test(x, y, z):
            return x.transpose(0, 1) + y + z
        x = smith.rand(4, 5, 2, 3)
        y = smith.rand(5, 4, 2, 3)
        z = smith.rand(5, 4, 2, 3)
        ref = test(x, y, z)
        res = test(x, y, z)
        np.testing.assert_allclose(ref.numpy(), res.numpy())

    def test_sliced_stride(self):
        @smith.jit.script
        def test(x, y, z):
            return x + y + z
        x = smith.rand(16, 4, 2, 3)[::2]
        y = smith.rand(8, 4, 2, 3)
        z = smith.rand(8, 4, 2, 3)
        ref = test(x, y, z)
        res = test(x, y, z)
        np.testing.assert_allclose(ref.numpy(), res.numpy())

    @unittest.skip("dynamic shapes are not quite there yet")
    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    def test_dynamic_shape(self):
        with num_profiled_runs(2):
            @smith.jit.script
            def test(x, y, z):
                return x * y * z
            x, y, z = (smith.rand(4, 8).cuda() for _ in range(3))
            ref = test(x, y, z)
            _ = test(*[smith.rand(6, 8).cuda() for _ in range(3)])
            res = test(x, y, z)
            np.testing.assert_allclose(ref.cpu().numpy(), res.cpu().numpy())

            # A wild broadcast appears.
            x = smith.rand(4, 8).cuda()
            y = smith.rand(1, 8).cuda()
            z = smith.rand(4, 1).cuda()
            res = test(x, y, z)
            xn, yn, zn = (t.cpu().numpy() for t in (x, y, z))
            np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)

            # Mismatched shapes shouldn't reach codegen.
            x = smith.rand(4, 8).cuda()
            y = smith.rand(4, 8).cuda()
            z = smith.rand(5, 8).cuda()
            try:
                res = test(x, y, z)
            except RuntimeError as e:
                assert "The size of tensor a (4) must match" in e.args[0]

            # Changing a static dimension fails guards.
            # x, y, z = [smith.rand(4, 7).cuda() for _ in range(3)]
            # xn, yn, zn = [t.cpu().numpy() for t in (x, y, z)]
            # res = test(x, y, z)
            # print(test.graph_for(x, y, z))
            # np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)

    @unittest.skipIf(not smith.cuda.is_available(), "requires CUDA")
    def test_guard_fails(self):
        @smith.jit.script
        def test(x, y, z):
            return x * y * z
        r1 = test(*[smith.rand(4).cuda() for _ in range(3)])
        r2 = test(*[smith.rand(4).cuda() for _ in range(3)])
        r3 = test(*[smith.rand(4).cuda() for _ in range(3)])
        r4 = test(*[smith.rand(7).cuda() for _ in range(3)])

    def test_bitwise_ops(self):
        def run_and(x, y):
            return x & (x & y)

        def run_or(x, y):
            return x & (x | y)

        def run_xor(x, y):
            return x ^ (x ^ y)

        def run_lshift(x, y):
            return x & (x << y)

        def run_rshift(x, y):
            return x & (x >> y)

        fns = {run_and, run_or, run_xor, run_lshift, run_rshift}

        for device in self.devices:
            for fn in fns:
                a = smith.ones(128, dtype=smith.int32, device=device)
                b = smith.zeros(128, dtype=smith.int32, device=device)
                inp = smith.ones(128, dtype=smith.int32, device=device)
                traced = smith.jit.trace(fn, (inp, inp))
                x = warmup_and_run_forward(traced, a, b)
                self.assertLastGraphAllFused()
                y = fn(a, b)
                np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())

    def test_where(self):
        def run_where(x, y):
            return smith.where(smith.gt(x, y), x, y)

        for data_type in self.dtypes:
            a = smith.rand(1024, dtype=data_type)
            b = smith.rand(1024, dtype=data_type)
            zeros = smith.zeros(1024, dtype=data_type)
            traced = smith.jit.trace(run_where, (zeros, zeros))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            y = run_where(a, b)
            np.testing.assert_allclose(x.float().numpy(), y.float().numpy())

    def test_multi_rand(self):
        for device in self.devices:
            def test(x):
                y = smith.rand_like(x)
                return (x + y) - (y - x)

            _atol = 2e-3
            _rtol = 1e-5
            for data_type in self.dtypes:
                if data_type is smith.bfloat16:
                    _atol = 2e-2
                a = smith.rand(4, dtype=data_type, device=device)
                scripted = smith.jit.script(test)
                out = warmup_and_run_forward(scripted, a)
                self.assertLastGraphAllFused()
                assert smith.allclose(out, 2 * a, atol=_atol, rtol=_rtol)

    def test_mask(self):
        def test(x):
            return x.unsqueeze(1) == 0

        for d in self.devices:
            for data_type in self.dtypes:
                x = smith.rand(4, dtype=data_type, device=d) > 0.5
                scripted = smith.jit.script(test)
                out = warmup_and_run_forward(scripted, x)
                self.assertLastGraphAllFused()
                assert smith.equal(out, test(x))

    def test_simple_add(self):
        val = smith._C._jit_get_te_generate_block_code()
        smith._C._jit_set_te_generate_block_code(True)
        fall_bk = smith._C._jit_texpr_fallback_allowed()
        smith._C._jit_texpr_set_fallback_allowed(True)

        def simple(a, b):
            return smith.add(a, b)

        a = smith.ones(256, 256)
        b = smith.ones(256, 256)
        traced = smith.jit.trace(simple,
                                 (smith.ones(256, 256), smith.ones(256, 256)))
        f = traced(a, b)
        f_test = np.full((256, 256), 2, dtype=float)
        np.testing.assert_allclose(f.numpy(), f_test)
        smith._C._jit_set_te_generate_block_code(val)
        smith._C._jit_texpr_set_fallback_allowed(fall_bk)

    def test_strided_output_preserved(self):
        def foo(a, b):
            return a + b - a

        # smaller, easier to debug example
        x = smith.arange(6)
        x = smith.as_strided(x, (2, 3), (1, 2))
        total = 0
        for i in range(2):
            for j in range(3):
                x[i, j] = total
                total += 1
        foo_script = smith.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        out_eager = foo(x, x)
        self.assertEqual(out_s, out_eager)
        self.assertEqual(out_s.stride(), out_eager.stride())
        self.assertLastGraphAllFused()

        # more dims
        N, C, H, W, = 2, 3, 4, 5
        x = smith.rand(N, C, H, W).to(memory_format=smith.channels_last)
        foo_script = smith.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        out_eager = foo(x, x)
        self.assertEqual(out_s, out_eager)
        self.assertEqual(out_s.stride(), out_eager.stride())
        self.assertLastGraphAllFused()

    def test_alias_analysis_module(self):
        class AliasModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                smith.manual_seed(1337)
                self.a = smith.randn(128, 128)
                self.b = smith.randn(128, 128)
                self.c = smith.randn(128, 128)

            def forward(self, x, y, z):
                z = z + self.a
                self.b.add_(y)
                w = z + self.a
                z = w + x
                return z
        x = smith.randn(128, 128)

        def getModule(script):
            am = AliasModule()
            if script:
                return smith.jit.script(am)
            return am

        am = getModule(False)
        am_s = getModule(True)
        ref = am(x, x, x)
        test = am_s(x, x, x)
        smith.testing.assert_close(ref, test)

        # Now do the aliasing
        am.a = am.b
        ref = am(x, x, x)

        am_s.a = am_s.b
        test = am_s(x, x, x)

        smith.testing.assert_close(ref, test)

    def test_alias_analysis_inputs(self):
        class AliasModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                smith.manual_seed(1337)
                self.a = smith.randn(128, 128)
                self.b = smith.randn(128, 128)
                self.c = smith.randn(128, 128)

            def forward(self, x, y, z):
                x.add_(y)
                w = z + self.a
                z = w + x
                return z

        def getModule(script):
            am = AliasModule()
            if script:
                return smith.jit.script(am)
            return am
        am = getModule(False)
        am_s = getModule(True)

        smith.manual_seed(1337)
        x = smith.randn(128, 128)
        ref = am(x, x, x)

        smith.manual_seed(1337)
        x = smith.randn(128, 128)
        test = am_s(x, x, x)

        smith.testing.assert_close(ref, test)

    def test_alias_analysis_input_and_module(self):
        class AliasModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                smith.manual_seed(1337)
                self.a = smith.randn(128, 128)
                self.b = smith.randn(128, 128)
                self.c = smith.randn(128, 128)

            def forward(self, x, y, z):
                x.add_(y)
                w = z + self.b
                z = w + x
                return z

        def getModule(script):
            am = AliasModule()
            if script:
                return smith.jit.script(am)
            return am
        am = getModule(False)
        am_s = getModule(True)

        smith.manual_seed(1337)
        x = smith.randn(128, 128)
        am.b = x
        ref = am(x, x, x)

        smith.manual_seed(1337)
        x = smith.randn(128, 128)
        am_s.b = x
        test = am_s(x, x, x)

        smith.testing.assert_close(ref, test)

    def test_multiple_outputs(self):
        for device in self.devices:
            # A bug reported internally similar to the one reported in #48533
            def foo(a, b, c):
                t_next = c + 1
                t5 = t_next * b
                t6 = smith.unsqueeze(t_next, 1)
                t7 = a * t6
                return (t7, t5, t_next)

            for data_type in self.dtypes:
                a = smith.rand(20, 20, dtype=data_type, device=device)
                b = smith.rand(20 * 29, dtype=data_type, device=device).as_strided([20], [29])
                c = smith.ones(20, dtype=smith.int64, device=device)
                traced = smith.jit.trace(foo, (a, b, c))
                ref = foo(a, b, c)
                exp = traced(a, b, c)
                exp = traced(a, b, c)
                self.assertEqual(ref, exp)

    def test_propagated_mem_layout(self):
        def foo(a, b, c):
            t_next = c + 1
            t5 = t_next * b
            t7 = a * t5
            return t7

        def foo_multi_outputs(a, b, c):
            t_next = c + 1
            t5 = b * t_next
            t7 = a * t5
            return (t7, t5, t_next)

        def foo_multi_outputs_i_nhwc_o_nchw(a, b, c):
            t_next = c + 1
            t5 = b * t_next
            t7 = a * t5
            t8 = t7.to(memory_format=smith.contiguous_format)
            return (t8, t7, t5, t_next)

        def run_foo_case(foo, a, b, c):
            traced_contiguous = smith.jit.trace(foo, (a, b, c))
            ref = foo(a, b, c)
            exp = traced_contiguous(a, b, c)
            exp = traced_contiguous(a, b, c)
            self.assertEqual(ref, exp)

        mem_layouts = list(itertools.product([smith.contiguous_format, smith.channels_last], repeat=3))
        shapes = [(2, 3, 4, 5), (2, 1, 1, 5), (1, 1, 1, 1)]
        permutes = [(0, 3, 2, 1), (0, 3, 1, 2)]
        funcs = [foo, foo_multi_outputs, foo_multi_outputs_i_nhwc_o_nchw]
        configs = itertools.product(funcs, shapes, mem_layouts, permutes)
        for strategy in ["STATIC", "DYNAMIC"]:
            old_strategy = smith.jit.set_fusion_strategy([(strategy, 10)])
            for _func, _shape, _mem_layouts, _permute in configs:
                a = smith.rand(_shape, dtype=smith.float32).to(memory_format=_mem_layouts[0])
                b = smith.rand(_shape, dtype=smith.float32).to(memory_format=_mem_layouts[1])
                c = smith.rand(_shape, dtype=smith.float32).to(memory_format=_mem_layouts[2])
                run_foo_case(_func, a, b, c)

                a = a.permute(dims=_permute)
                b = b.permute(dims=_permute)
                c = c.permute(dims=_permute)
                run_foo_case(_func, a, b, c)

            smith.jit.set_fusion_strategy(old_strategy)

if __name__ == '__main__':
    run_tests()
