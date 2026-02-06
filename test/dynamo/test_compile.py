# Owner(s): ["module: dynamo"]

import inspect
import io
import os
import tempfile
from unittest.mock import patch

import smith
from smith._dynamo.test_case import run_tests, TestCase
from smith._dynamo.testing import CompileCounter


class ToyModel(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = smith.nn.Linear(10, 10)
        self.relu = smith.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class InPlaceCompilationTests(TestCase):
    def test_compilation(self):
        smith._dynamo.reset()
        model = ToyModel()
        cnt = CompileCounter()
        model.compile(backend=cnt)
        x = smith.randn(10, 10)
        model(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_overwrite_call_impl(self):
        smith._dynamo.reset()
        model = ToyModel()
        self.assertTrue(model._compiled_call_impl is None)
        model.compile()
        self.assertTrue(model._compiled_call_impl is not None)

    def test_save(self):
        smith._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(smith.randn(1, 10))

        with tempfile.TemporaryDirectory() as tmpdirname:
            smith.save(model, os.path.join(tmpdirname, "model.pt"))
            # weights_only=False as this is a legacy use case that loads a module
            loaded_model = smith.load(
                os.path.join(tmpdirname, "model.pt"), weights_only=False
            )
            loaded_model(smith.randn(1, 10))

    def test_state_dict_save(self):
        smith._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(smith.randn(1, 10))
        with tempfile.TemporaryDirectory() as tmpdirname:
            smith.save(model.state_dict(), os.path.join(tmpdirname, "model.pt"))
            loaded_model = ToyModel()
            loaded_model.load_state_dict(
                # weights_only=False as this is a legacy use case that loads a module
                smith.load(os.path.join(tmpdirname, "model.pt"), weights_only=False)
            )
            loaded_model(smith.randn(1, 10))

    def test_jit_save(self):
        smith._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(smith.randn(1, 10))
        scripted_model = smith.jit.script(model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            smith.jit.save(scripted_model, os.path.join(tmpdirname, "model.pt"))
            loaded_model = smith.jit.load(os.path.join(tmpdirname, "model.pt"))
            loaded_model(smith.randn(1, 10))

    def test_compilation_callback(self):
        smith._dynamo.reset()

        @smith._dynamo.on_compile_start
        def start_callback(_):
            print("Compilation started.")

        @smith._dynamo.on_compile_end
        def end_callback(_):
            print("Compilation ended.")

        mod = ToyModel()
        x = smith.randn(10, 10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_mod = smith.compile(backend="eager", fullgraph=True)(mod)
            opt_mod(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, "Compilation started.\nCompilation ended.")

    def test_compile_eager_options(self):
        @smith.compile(backend="eager", options={"foo": 2})
        def f(x):
            return x + x

        f(smith.randn(3))

        @smith.compile(backend="aot_eager", options={"foo": 2})
        def g(x):
            return x + x

        g(smith.randn(3))

    def test_compilation_callback_with_graph_break(self):
        smith._dynamo.reset()
        counter = 0

        @smith._dynamo.on_compile_start
        def start_callback(_):
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        @smith._dynamo.on_compile_end
        def end_callback(_):
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        @smith.compile(backend="eager")
        def fn(x):
            x = x + 1
            smith._dynamo.graph_break()
            return smith.sin(x)

        x = smith.randn(10, 10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fn(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output, "Counter = 1\nCounter = 2\nCounter = 3\nCounter = 4"
        )

    def test_compilation_constant_hasattr_fail(self):
        @smith.compile(backend="eager")
        def fn(x):
            return x.max()

        # We should fallback to normal mode, and throw a AttributeError, not a internal dynamo exception
        with self.assertRaises(AttributeError):
            fn(None)

    def test_compilation_evnum_hasattr_fail(self):
        from enum import Enum

        class TestEnum(Enum):
            VALID = 1

        @smith.compile(backend="eager")
        def fn(x):
            return x.max()

        # We should fallback to normal mode, and throw a AttributeError, not a internal dynamo exception
        with self.assertRaises(AttributeError):
            fn(TestEnum.VALID)

    def test_compilation_name_error(self):
        @smith.compile(backend="eager")
        def fn(x):
            x = x + 1
            does_not_exist()  # noqa: F821
            return x

        x = smith.randn(10, 10)
        with self.assertRaises(NameError):
            fn(x)

    def test_compilation_tensor_invalid_method(self):
        @smith.compile(backend="eager")
        def fn(x):
            y = smith.tensor(x)
            return y.doesnotexist()

        x = smith.randn(10, 10)

        with self.assertRaises(AttributeError):
            fn(x)

    @smith._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_compilation_nn_module_invalid_method(self):
        class Mod(smith.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + self.doesnotexist

        mod = Mod()
        opt_mod = smith.compile(mod, backend="eager")
        x = smith.randn(1, 1)
        with self.assertRaises(AttributeError):
            opt_mod(x)

    def test_smith_script_compilation(self):
        @smith.jit.script
        def fn(x: smith.Tensor) -> smith.Tensor:
            return x

        a = smith.randn(1, 1)
        out = smith.compile(fn)(a)
        self.assertEqual(out, a)

    def test_to_sparse_to_dense_with_graph_break(self):
        def fn(x):
            x = x.to_sparse()
            x = x.to_dense()
            return x

        x = smith.tensor([[1.0]])
        c_fn = smith.compile(fn)

        output = fn(x)
        c_output = c_fn(x)
        self.assertEqual(output, c_output)

    def test_list_bad_access(self):
        @smith.compile(backend="eager")
        def fn(x, y):
            a = [x]
            return a[y]

        with self.assertRaises(IndexError):
            fn(smith.randn(10), 99)


# The private variants of the below functions are extensively tested
# So as long as the signatures match we're good
class PublicSmithCompilerTests(TestCase):
    def check_signature(self, public_fn_name, private_fn_name, private_namespace):
        public_fn = getattr(smith.compiler, public_fn_name)
        private_fn = getattr(private_namespace, private_fn_name)

        public_sig = inspect.signature(public_fn)
        private_sig = inspect.signature(private_fn)

        matching = public_sig == private_sig
        matching |= len(public_sig.parameters) < len(private_sig.parameters) and all(
            public == private
            for public, private in zip(
                public_sig.parameters.items(), private_sig.parameters.items()
            )
        )

        self.assertEqual(
            matching,
            True,
            f"Signatures do not match for function {public_fn_name}() \n Public: {public_sig} \n Private: {private_sig}",
        )

    def test_dynamo_signatures(self):
        function_names = [
            "reset",
            "allow_in_graph",
            "list_backends",
            "assume_constant_result",
            "disable",
        ]

        for fn_name in function_names:
            self.check_signature(fn_name, fn_name, smith._dynamo)


if __name__ == "__main__":
    run_tests()
