# Owner(s): ["module: dynamo"]

import unittest

import smith
import smith._dynamo
import smith._dynamo.config
import smith._dynamo.test_case
from smith._dynamo.comptime import comptime
from smith._dynamo.exc import Unsupported
from smith.testing._internal.common_device_type import skipIf
from smith.testing._internal.common_utils import (
    IS_FBCODE,
    munge_exc,
    skipIfWindows,
    TEST_Z3,
)
from smith.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class ExcTests(LoggingTestCase):
    maxDiff = None

    def test_unsupported_real_stack(self):
        # exercise Unsupported constructor and augment_exc_message
        def fn002(x):
            smith._dynamo.graph_break()

        def fn001(x):
            x = x + 1
            fn002(x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: smith.compile(fn001, backend="eager", fullgraph=True)(
                smith.randn(1)
            ),
            """\
Call to `smith._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `smith._dynamo.graph_break()` call.

  Developer debug context: Called `smith._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-blacksmith.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_exc.py", line N, in fn001
    fn002(x)
  File "test_exc.py", line N, in fn002
    smith._dynamo.graph_break()""",
        )

    @smith._dynamo.config.patch(verbose=True, suppress_errors=True)
    @make_logging_test()
    @unittest.skipIf(IS_FBCODE, "stack trace slightly different in fbcode")
    def test_internal_error_suppress_errors(self, records):
        def fn001(x):
            def f(ctx):
                raise AssertionError

            comptime(f)

        smith.compile(fn001, backend="eager")(smith.randn(1))

        record = self.getRecord(records, "WON'T CONVERT")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
========== SmithDynamo Stack Trace ==========
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise AssertionError
AssertionError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)


========== The above exception occurred while processing the following code ==========

  File "test_exc.py", line N, in test_internal_error_suppress_errors
    smith.compile(fn001, backend="eager")(smith.randn(1))
  File "test_exc.py", line N, in fn001
    comptime(f)

==========""",
        )

    @make_logging_test()
    def test_not_implemented_error(self, records):
        def fn001(x):
            def f(ctx):
                raise NotImplementedError

            # Ensure graph break is not possible
            for _ in range(3):
                comptime(f)

        smith.compile(fn001, backend="eager")(smith.randn(1))

        record = self.getRecord(records, "WON'T CONVERT")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
due to:
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise NotImplementedError
smith._dynamo.exc.InternalSmithDynamoError: NotImplementedError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

    @smith._dynamo.config.patch(inject_BUILD_SET_unimplemented_TESTING_ONLY=True)
    @make_logging_test(graph_breaks=True)
    def test_unsupported_error(self, records):
        def fn001(x):
            return {1, 2}

        smith.compile(fn001, backend="eager")(smith.randn(1))

        record = self.getRecord(records, "missing BUILD_SET handler")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
Graph break in user code at test_exc.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

missing BUILD_SET handler
  Explanation: Missing BUILD_SET bytecode handler (for testing purposes).


  Developer debug context:

 For more details about this graph break, please visit: https://meta-blacksmith.github.io/compile-graph-break-site/gb/gb0200.html

User code traceback:
  File "test_exc.py", line N, in test_unsupported_error
    smith.compile(fn001, backend="eager")(smith.randn(1))
  File "test_exc.py", line N, in fn001
    return {1, 2}
""",  # noqa: B950
        )

    @smith._dynamo.config.patch(suppress_errors=False)
    def test_internal_error_no_suppress(self):
        def fn001(x):
            # NB: avoid decorator, as 3.11 changed the line number attributed
            # in this situation
            def f(ctx):
                raise AssertionError

            comptime(f)

        # NB: OK for user code to be truncated here, because the regular
        # exception backtrace has the rest of the crumbs
        self.assertExpectedInlineMunged(
            AssertionError,
            lambda: smith.compile(fn001, backend="eager")(smith.randn(1)),
            """\


from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log(self, records):
        def fn002(x):
            x = x + 1
            smith._dynamo.graph_break()
            x = x + 1
            return x

        def fn001(x):
            return fn002(x)

        smith.compile(fn001, backend="eager")(smith.randn(1))

        record = self.getRecord(records, "Graph break in user code")

        # TODO: This should also report the enclosing frames; need to plumb
        # frame object to it
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
Graph break in user code at test_exc.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `smith._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `smith._dynamo.graph_break()` call.

  Developer debug context: Called `smith._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-blacksmith.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_exc.py", line N, in test_graph_break_log
    smith.compile(fn001, backend="eager")(smith.randn(1))
  File "test_exc.py", line N, in fn001
    return fn002(x)
  File "test_exc.py", line N, in fn002
    smith._dynamo.graph_break()
""",  # noqa: B950
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log_generic_jump(self, records):
        def fn(x):
            if x.sum() > 0:
                return x + 1
            else:
                return x - 1

        smith.compile(fn, backend="eager")(smith.ones(3, 3))

        # check for record existence
        self.getRecord(records, "Graph break in user code")

    @smith._dynamo.config.patch(suppress_errors=False)
    def test_backend_suppress_line(self):
        def fn001(x):
            x = smith.relu(x)
            return x + 1

        # Do NOT let this get attributed to x + 1
        self.assertExpectedInlineMunged(
            smith._dynamo.exc.BackendCompilerFailed,
            lambda: smith.compile(fn001, backend="relu_compile_error_TESTING_ONLY")(
                smith.randn(1)
            ),
            """\
backend='relu_compile_error_TESTING_ONLY' raised:
ReluCompileError:""",
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    @smith._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    @smith.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
        translation_validation_no_bisect=True,
    )
    @skipIfWindows(
        msg='AssertionError: "tran[551 chars]s1 s2 s3) s0)\n  ==> (<= (+ s1 s2) (+ s0 (* -1[511 chars][0])'  # noqa: PLR0133
        != 'tran[551 chars]s1 s2) (+ s0 (* -1 s3)))\n  ==> (<= (+ s1 s2) [483 chars][0])"'
    )
    def test_trigger_on_error(self):
        from smith.fx.experimental.validator import ValidationException

        @smith.compile
        def fn(x, shape):
            return x.split(shape)

        self.assertExpectedInlineMunged(
            ValidationException,
            lambda: fn(smith.randn(20), (5, 10, 5)),
            """\
translation validation failed.

Model:
  ==> L['shape'][0]: 0
  ==> L['shape'][1]: 0
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s3: 0
  ==> s52: 0
  ==> s77: 3
  ==> s86: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 1)

Target Expressions:
  ==> (!= (+ s3 s52 s86) s77)
  ==> (<= 0 s3)
  ==> (<= 0 s52)
  ==> (<= 0 s86)
  ==> (<= 2 s77)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 0)
  ==> (>= 0 s86)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    @smith._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    @smith.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
    )
    def test_trigger_bisect_on_error(self):
        from smith.fx.experimental.validator import BisectValidationException

        @smith.compile
        def fn(x, shape):
            return x.split(shape)

        self.assertExpectedInlineMunged(
            BisectValidationException,
            lambda: fn(smith.randn(20), (5, 10, 5)),
            """\
translation validation failed when evaluating: Eq(s3 + s52 + s86, s77)

Failure occurred while running node:
    %split : [num_users=3] = call_method[target=split](args = (%l_x_, (%l_shape_0_, %l_shape_1_, %l_shape_2_)), kwargs = {})

Model:
  ==> L['shape'][0]: 0
  ==> L['shape'][1]: 0
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s3: 0
  ==> s52: 0
  ==> s77: 3
  ==> s86: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 1)

Target Expressions:
  ==> (!= (+ s3 s52 s86) s77)
  ==> (<= 0 s3)
  ==> (<= 0 s52)
  ==> (<= 0 s86)
  ==> (<= 2 s77)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 0)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )


if __name__ == "__main__":
    from smith._dynamo.test_case import run_tests

    run_tests()
