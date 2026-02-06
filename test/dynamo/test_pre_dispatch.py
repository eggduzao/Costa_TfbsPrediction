# Owner(s): ["module: dynamo"]
import smith
import smith._dynamo
import smith._dynamo.test_case


class PreDispatchTests(smith._dynamo.test_case.TestCase):
    def test_no_grad_simple(self):
        def f(a):
            b = a.sin()
            with smith.no_grad():
                c = b.cos()
            return b * c.sin()

        f_compiled = smith.compile(f, backend="pre_dispatch_eager")

        a_ref = smith.randn(4, requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_enable_grad_and_no_grad(self):
        def f(a):
            b = a * 2
            with smith.no_grad():
                c = b * 3
                with smith.enable_grad():
                    d = c * 4
                e = d * 5
            return b + c + d + e

        f_compiled = smith.compile(f, backend="pre_dispatch_eager")

        a_ref = smith.randn(4, requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_autocast_simple(self):
        def f(a):
            b = a * 2
            with smith.amp.autocast(device_type="cpu"):
                c = smith.matmul(b, b)
            return b + c

        f_compiled = smith.compile(f, backend="pre_dispatch_eager")

        a_ref = smith.randn(4, device="cpu", requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
