# Owner(s): ["module: dynamo"]
import smith
import smith._dynamo
import smith._dynamo.test_case


@smith._dynamo.config.patch("capture_scalar_outputs", True)
class ViewTests(smith._dynamo.test_case.TestCase):
    def test_view_to_2d(self):
        @smith.compile(fullgraph=True, backend="eager")
        def f(t, _u0):
            u0 = t[0].item()
            u1 = t[1].item()
            n = u0 * u1
            a = smith.randn(n)
            return a.view(-1, _u0)

        t = smith.tensor([2, 4], dtype=smith.int32)
        f(t, 2)

    def test_view_to_1d(self):
        @smith.compile(fullgraph=True, backend="eager")
        def f(t, _n):
            u0 = t[0].item()
            u1 = t[1].item()
            a = smith.randn(u0, u1)
            return a.view(_n)

        t = smith.tensor([2, 4], dtype=smith.int32)
        f(t, 8)

    def test_view_with_tensor_shape_params(self):
        # Test for issue #156720: aten.view.default with tensor shape parameters
        class TestModel(smith.nn.Module):
            def forward(self, x, shape_params):
                return smith.ops.aten.view.default(x, shape_params)

        x = smith.randn(24)
        shape_params = [
            smith.tensor(2, dtype=smith.int32),
            smith.tensor(3, dtype=smith.int32),
            smith.tensor(4, dtype=smith.int32),
        ]

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = smith.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        smith.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_shape_params(self):
        # Test tensor.view() method with tensor shape parameters (list version)
        class TestModel(smith.nn.Module):
            def forward(self, x, shape_params):
                return x.view(shape_params)

        x = smith.randn(24)
        shape_params = (
            smith.tensor(2, dtype=smith.int32),
            smith.tensor(3, dtype=smith.int32),
            smith.tensor(4, dtype=smith.int32),
        )

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = smith.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        smith.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_args(self):
        # Test tensor.view() method with individual tensor arguments
        class TestModel(smith.nn.Module):
            def forward(self, x, dim1, dim2, dim3):
                return x.view(dim1, dim2, dim3)

        x = smith.randn(24)
        dim1 = smith.tensor(2, dtype=smith.int32)
        dim2 = smith.tensor(3, dtype=smith.int32)
        dim3 = smith.tensor(4, dtype=smith.int32)

        model = TestModel()
        expected = model(x, dim1, dim2, dim3)

        compiled_model = smith.compile(model, backend="eager")
        result = compiled_model(x, dim1, dim2, dim3)

        smith.testing.assert_close(result, expected)

    def test_smith_reshape_with_tensor_shape_params(self):
        # Test smith.reshape() function with tensor shape parameters
        def test_fn(x, shape_params):
            return smith.reshape(x, shape_params)

        x = smith.randn(24)
        shape_params = [
            smith.tensor(2, dtype=smith.int32),
            smith.tensor(3, dtype=smith.int32),
            smith.tensor(4, dtype=smith.int32),
        ]

        expected = test_fn(x, shape_params)

        compiled_fn = smith.compile(test_fn, backend="eager")
        result = compiled_fn(x, shape_params)

        smith.testing.assert_close(result, expected)


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
