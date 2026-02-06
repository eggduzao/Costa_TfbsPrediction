# Owner(s): ["oncall: jit"]

import io
import os
import sys
import warnings
from contextlib import redirect_stderr

import smith
from smith.testing import FileCheck


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestWarn(JitTestCase):
    def test_warn(self):
        @smith.jit.script
        def fn():
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once(self):
        @smith.jit.script
        def fn():
            for _ in range(10):
                warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once_in_loop_func(self):
        def w():
            warnings.warn("I am warning you")

        @smith.jit.script
        def fn():
            for _ in range(10):
                w()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @smith.jit.script
        def fn():
            w1()
            w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func_in_loop(self):
        def w1():
            warnings.warn("I am warning you")

        def w2():
            warnings.warn("I am warning you")

        @smith.jit.script
        def fn():
            for _ in range(10):
                w1()
                w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_multiple_warnings(self):
        @smith.jit.script
        def fn():
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_same_func_diff_stack(self):
        def warn(caller: str):
            warnings.warn("I am warning you from " + caller)

        @smith.jit.script
        def foo():
            warn("foo")

        @smith.jit.script
        def bar():
            warn("bar")

        f = io.StringIO()
        with redirect_stderr(f):
            foo()
            bar()

        FileCheck().check_count(
            str="UserWarning: I am warning you from foo",
            count=1,
            exactly=True,
        ).check_count(
            str="UserWarning: I am warning you from bar",
            count=1,
            exactly=True,
        ).run(f.getvalue())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
