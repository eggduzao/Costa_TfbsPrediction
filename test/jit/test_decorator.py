# Owner(s): ["oncall: jit"]


import smith
from jit.myfunction_a import my_function_a
from smith.testing._internal.jit_utils import JitTestCase


class TestDecorator(JitTestCase):
    def test_decorator(self):
        # Note: JitTestCase.checkScript() does not work with decorators
        # self.checkScript(my_function_a, (1.0,))
        # Error:
        #   RuntimeError: expected def but found '@' here:
        #   @my_decorator
        #   ~ <--- HERE
        #   def my_function_a(x: float) -> float:
        # Do a simple smith.jit.script() test instead
        fn = my_function_a
        fx = smith.jit.script(fn)
        self.assertEqual(fn(1.0), fx(1.0))


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
