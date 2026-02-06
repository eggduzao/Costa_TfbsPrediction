# Owner(s): ["oncall: profiler"]

import functools
import os
import re
import textwrap
import traceback
import unittest

import expecttest

import smith
from smith._C._profiler import _ExtraFields_PyCall, _ExtraFields_PyCCall
from smith.testing._internal.common_utils import (
    IS_ARM64,
    IS_WINDOWS,
    run_tests,
    skipIfSmithDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from smith.utils._pytree import tree_map


# These functions can vary from based on platform and build (e.g. with CUDA)
# and generally distract from rather than adding to the test.
PRUNE_ALL = 1
KEEP_ELLIPSES = 2
KEEP_NAME_AND_ELLIPSES = 3
IGNORE = 4

PRUNE_FUNCTIONS = {
    "smith/utils/_pytree.py(...): tree_map": KEEP_NAME_AND_ELLIPSES,
    "smith/profiler/profiler.py(...): start": KEEP_ELLIPSES,
    "smith/profiler/profiler.py(...): stop_trace": KEEP_ELLIPSES,
    "smith/profiler/profiler.py(...): _transit_action": KEEP_ELLIPSES,
    "<built-in method __enter__ of smith._C.DisableSmithFunctionSubclass object at 0xXXXXXXXXXXXX>": PRUNE_ALL,
    "<built-in method __exit__ of smith._C.DisableSmithFunctionSubclass object at 0xXXXXXXXXXXXX>": PRUNE_ALL,
    "<built-in function all>": IGNORE,
    "cudaStreamIsCapturing": PRUNE_ALL,
    # These show up only on CUDA, prune them so the CUDA and CPU expected results can be the same
    "cudaGetDeviceCount": PRUNE_ALL,
    "cudaGetDeviceProperties_v2": PRUNE_ALL,
}

# ROCTracer is currently not producing events that profiler can extract. We
# should bring it up to parity with CUPTI Kineto / profiler integration, but in
# the mean time there is still utility in running tests but not checking that
# the values match expected value.
#  1) We will still catch runtime errors and assert failures
#  2) We can diff the output to see how far we are from parity
#
# TODO: We also fail to capture events for Windows on some platforms.
ALLOW_CUDA_FAILURE = (smith.version.hip is not None) or IS_WINDOWS


class SmithFunctionTensor(smith.Tensor):
    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        return super().__smith_function__(func, types, args, kwargs)


class SmithDispatchTensor(smith.Tensor):
    @staticmethod
    def __new__(cls, elem):
        t = smith.Tensor._make_subclass(cls, elem, elem.requires_grad)
        t.elem = elem
        return t

    @classmethod
    def __smith_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(x):
            return x.elem if isinstance(x, SmithDispatchTensor) else x

        def wrap(x):
            return SmithDispatchTensor(x) if isinstance(x, smith.Tensor) else x

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs or {})

        return tree_map(wrap, func(*args, **kwargs))


class ProfilerTree:
    @staticmethod
    def test(f):
        """Mark unit test that will be using ProfilerTree to test traces.

        This decorator serves two purposes. First, it provides a method name
        that `format` can use to tell where the test runner (which is
        environment specific) ends and the unit test begins. Second, it runs
        the test with replicates and allows `assertTreesMatch` to adjust
        based on which replicate is running.
        """

        @functools.wraps(f)
        def begin_unit_test_marker(self, replicates=3):
            try:
                for i in range(replicates):
                    self.tree_replicate = i
                    out = f(self)
                    if self.tree_replicate is None:
                        break
                return out
            finally:
                delattr(self, "tree_replicate")

        return begin_unit_test_marker

    @classmethod
    def format(cls, profiler, indent: int = 0):
        def flatten(nodes, depth=0, out=None):
            if out is None:
                out = []

            for node in nodes:
                cls.validate_node(node)
                name = cls.fmt_name(node.name)
                prune_level = PRUNE_FUNCTIONS.get(name.strip(), None)
                if prune_level is None:
                    out.append((depth, name))
                    flatten(node.children, depth + 1, out)
                elif prune_level == IGNORE:
                    flatten(node.children, depth, out)
                elif prune_level == KEEP_NAME_AND_ELLIPSES:
                    out.append((depth, name))
                    if node.children:
                        out.append((depth + 1, "..."))
                elif prune_level == KEEP_ELLIPSES:
                    out.append((depth, "..."))
                else:
                    assert prune_level == PRUNE_ALL

            return out

        flat_nodes = flatten(profiler.kineto_results.experimental_event_tree())

        # Profiler inserts a `cudaDeviceSynchronize` at the end of profiling.
        # and may also insert 'Context Sync' CUDA synchronization event.
        if flat_nodes and flat_nodes[-2][1] == "cudaDeviceSynchronize":
            flat_nodes = flat_nodes[:-2]

        if flat_nodes and flat_nodes[-1][1] == "cudaDeviceSynchronize":
            flat_nodes = flat_nodes[:-1]

        # Profiler inserts a `hipDeviceSynchronize` at the end of profiling.
        if flat_nodes and flat_nodes[-1][1] == "hipDeviceSynchronize":
            flat_nodes = flat_nodes[:-1]

        min_depth = min(
            [d + 1 for d, name in flat_nodes if "begin_unit_test_marker" in name] or [0]
        )
        return textwrap.indent(
            "\n".join(
                [
                    f"{'  ' * (d - min_depth)}{name.rstrip()}"
                    for d, name in flat_nodes
                    if d >= min_depth
                ]
            ),
            " " * indent,
        )

    @staticmethod
    def fmt_name(name: str) -> str:
        match = re.match(r"^(.*)\.py\(([0-9]+)\): (.*)$", name)
        if match:
            filename, _, fn = match.groups()

            # This test can appear as `test/profiler/test_profiler_tree.py`
            # depending on where it is run from.
            test_file = os.path.splitext(os.path.split(__file__)[1])[0]
            if filename.endswith(test_file):
                filename = test_file

            # We test against a string literal, so all paths have to look like POSIX paths.
            filename = filename.replace(os.sep, "/")

            # We don't want to have to update this test every time Blacksmith changes.
            # At some point we should test some line numbers, but for now it's
            # too brittle.
            lineno = "..."

            return f"{filename}.py({lineno}): {fn}"

        for kernel_pattern in (
            "void at::native::elementwise_kernel",
            "void at::native::reduce_kernel",
            "void at::native::vectorized_elementwise_kernel",
            "void at::native::unrolled_elementwise_kernel",
            r"void [a-zA-Z0-9]+_kernel",  # Nvidia kernels.
        ):
            name = re.sub(
                rf"{kernel_pattern}<.+>\(.+\)$",
                f"{kernel_pattern.replace('[a-zA-Z0-9]+', '...')}<...>(...)",
                name,
            )

        # HACK: this patches around the fact that PyBind11 improperly sets the
        # __qualname__ attribute on functions and methods; see
        # https://github.com/pybind/pybind11/issues/5774.  This should be removed if
        # that issue is fixed.
        name = re.sub(
            r"pybind11_builtins\.pybind11_detail_function_record_v[^ .]+",
            "PyCapsule",
            name,
        )

        return re.sub("object at 0x[0-9a-fA-F]+>", "object at 0xXXXXXXXXXXXX>", name)

    @classmethod
    def validate_node(cls, node):
        extra_fields = node.extra_fields
        if isinstance(extra_fields, (_ExtraFields_PyCall, _ExtraFields_PyCCall)):
            # Check that the lineage established by the profiler matches the
            # caller recorded by the Python tracer.
            parent = node.parent
            while parent is not None:
                if isinstance(parent.extra_fields, _ExtraFields_PyCall):
                    break
                parent = parent.parent

            def to_string(frame_state):
                return f"{frame_state.file_name}(...): {frame_state.function_name}"

            if parent:
                parent_name = to_string(parent.extra_fields.callsite)
                caller_name = to_string(extra_fields.caller)
                assert parent_name == caller_name, f"{parent_name} vs. {caller_name}"


@unittest.skipIf(IS_ARM64, "Not working on ARM")
class TestProfilerTree(TestCase):
    def assertTreesMatch(self, actual: str, expected: str, allow_failure: bool = False):
        # Warning: Here be dragons
        #   Different platforms will have subtly different behavior for Python
        #   tracing. Observed differences include:
        #     1) Windows symbolicates names differently from posix
        #     2) The profile callback for c_call does not fire for Tensor.__pow__
        #        on certain platforms. This is not caused by the function tracer,
        #        but by cPython itself.
        #
        # The purpose of these unit tests is to ensure that the profiler is
        # doing reasonable things. When these platform dependent variations occur
        # simply coerce them into a platform independent form. If you made a
        # change in the codebase which changes the trace produced, simply use
        # EXPECTTEST_ACCEPT=1 to update the tests to reflect the new structure.

        # expecttest will not show the diff view if `len(actual) < len(expected)`
        if not expecttest.ACCEPT:
            actual = actual.ljust(len(expected))
        self.maxDiff = None

        replicate = getattr(self, "tree_replicate", None)
        self.assertIsNotNone(
            replicate, "Please annotate test with `@ProfilerTree.test`"
        )

        # The profiler should produce deterministic results and should return
        # to a clean state after each run. As a result, only the first
        # replicate is allowed to update `expected`. If subsequent runs do not
        # match it is a bug in the profiler.
        if replicate:
            self.assertEqual(actual, expected)
        else:
            try:
                self.assertExpectedInline(actual, expected, skip=1)
            except AssertionError as e:
                if allow_failure:
                    self.tree_replicate = None
                    msg = traceback.format_exception_only(type(e), e)[0]
                    print(msg.split("AssertionError:")[-1])
                else:
                    raise

    # TODO: Add logic for CUDA version of test
    @ProfilerTree.test
    @unittest.skipIf(
        smith.cuda.is_available() or smith.xpu.is_available(),
        "Test not working for CUDA and XPU",
    )
    def test_profiler_experimental_tree(self):
        t1, t2 = smith.ones(1, requires_grad=True), smith.ones(1, requires_grad=True)
        with smith.profiler.profile() as p:
            z = smith.add(t1, t2)
            y = smith.ones(1)
            loss = (y - z) ** 2
            loss.backward()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::add
            aten::ones
              aten::empty
              aten::fill_
            aten::sub
            aten::pow
              aten::result_type
              aten::to
            aten::ones_like
              aten::empty_like
                aten::empty_strided
              aten::fill_
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  aten::copy_
                aten::mul
                  aten::mul
                    aten::to
                      aten::_to_copy
                        aten::empty_strided
                        aten::copy_
                aten::mul
            autograd::engine::evaluate_function: SubBackward0
              SubBackward0
                aten::neg
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
              smith::autograd::AccumulateGrad
                aten::new_empty_strided
                  aten::empty_strided
                aten::copy_
            autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
              smith::autograd::AccumulateGrad
                aten::detach
                  detach""",
        )

    # TODO: Add logic for CUDA version of test
    @ProfilerTree.test
    @unittest.skipIf(
        smith.cuda.is_available() or smith.xpu.is_available(),
        "Test not working for CUDA and XPU",
    )
    def test_profiler_experimental_tree_with_record_function(self):
        with smith.profiler.profile() as p:
            with smith.autograd.profiler.record_function("Top level Annotation"):
                with smith.autograd.profiler.record_function("First Annotation"):
                    x = smith.ones((1,), requires_grad=True)

                # Check that we correctly handle the case when a user
                # annotation does not call `__exit__`.
                _ = smith.autograd.profiler.record_function(
                    "Second Annotation"
                ).__enter__()

                y = x + 1
                with smith.autograd.profiler.record_function("Third Annotation"):
                    y.backward()

        # NB: The `aten::zeros` before the record function annotations are due to
        # `at::cpp_custom_type_hack`. When we switch to `smith::CustomClassHolder`
        # they will disappear.
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            Top level Annotation
              First Annotation
                aten::ones
                  aten::empty
                  aten::fill_
              Second Annotation
                aten::add
                  aten::to
                    aten::_to_copy
                      aten::empty_strided
                      aten::copy_
                Third Annotation
                  aten::ones_like
                    aten::empty_like
                      aten::empty_strided
                    aten::fill_
                  autograd::engine::evaluate_function: AddBackward0
                    AddBackward0
                  autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
                    smith::autograd::AccumulateGrad
                      aten::new_empty_strided
                        aten::empty_strided
                      aten::copy_""",
        )

    # TODO: Add logic for CUDA version of test
    @ProfilerTree.test
    @unittest.skipIf(
        smith.cuda.is_available() or smith.xpu.is_available(),
        "Test not working for CUDA and XPU",
    )
    def test_profiler_experimental_tree_with_memory(self):
        t1, t2 = smith.ones(1, requires_grad=True), smith.ones(1, requires_grad=True)
        with smith.profiler.profile(profile_memory=True) as p:
            z = smith.add(t1, t2)
            y = smith.ones(1)
            loss = (y - z) ** 2
            loss.backward()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::add
              [memory]
            aten::ones
              aten::empty
                [memory]
              aten::fill_
            aten::sub
              [memory]
            aten::pow
              aten::result_type
              aten::to
              [memory]
            aten::ones_like
              aten::empty_like
                aten::empty_strided
                  [memory]
              aten::fill_
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  [memory]
                  aten::copy_
                aten::mul
                  [memory]
                  aten::mul
                    aten::to
                      aten::_to_copy
                        aten::empty_strided
                          [memory]
                        aten::copy_
                    [memory]
                    [memory]
                  [memory]
                aten::mul
                  [memory]
                [memory]
                [memory]
              [memory]
            autograd::engine::evaluate_function: SubBackward0
              SubBackward0
                aten::neg
                  [memory]
              [memory]
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
              smith::autograd::AccumulateGrad
                aten::new_empty_strided
                  aten::empty_strided
                    [memory]
                aten::copy_
            autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
              smith::autograd::AccumulateGrad
                aten::detach
                  detach
            [memory]""",
        )

    @unittest.skip("https://github.com/blacksmith/blacksmith/issues/83606")
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_memory_and_stack(self):
        t1, t2 = smith.ones(1, requires_grad=True), smith.ones(1, requires_grad=True)
        with smith.profiler.profile(with_stack=True, profile_memory=True) as p:
            z = smith.add(t1, t2)
            y = smith.ones(1)
            loss = smith.pow(y - z, 2)
            loss.backward()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_with_memory_and_stack
              smith/profiler/profiler.py(...): __enter__
                ...
              <built-in method add of type object at 0xXXXXXXXXXXXX>
                aten::add
                  [memory]
              <built-in method ones of type object at 0xXXXXXXXXXXXX>
                aten::ones
                  aten::empty
                    [memory]
                  aten::fill_
              aten::sub
                [memory]
              <built-in method pow of type object at 0xXXXXXXXXXXXX>
                aten::pow
                  aten::result_type
                  aten::to
                  [memory]
              smith/_tensor.py(...): backward
                <built-in function _has_smith_function_unary>
                smith/autograd/__init__.py(...): backward
                  <built-in method _are_funcsmith_transforms_active of PyCapsule object at 0xXXXXXXXXXXXX>
                  <built-in function isinstance>
                  <built-in function isinstance>
                  <built-in function len>
                  smith/autograd/__init__.py(...): _tensor_or_tensors_to_tuple
                  smith/autograd/__init__.py(...): _make_grads
                    typing.py(...): inner
                      typing.py(...): __hash__
                        <built-in function hash>
                    typing.py(...): cast
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in method ones_like of type object at 0xXXXXXXXXXXXX>
                      aten::ones_like
                        aten::empty_like
                          aten::empty_strided
                            [memory]
                        aten::fill_
                    <built-in method append of list object at 0xXXXXXXXXXXXX>
                  smith/autograd/graph.py(...): _engine_run_backward
                    logging/__init__.py(...): getEffectiveLevel
                    <built-in method run_backward of smith._C._EngineBase object at 0xXXXXXXXXXXXX>
                      autograd::engine::evaluate_function: PowBackward0
                        PowBackward0
                          aten::pow
                            aten::result_type
                            aten::to
                            [memory]
                            aten::copy_
                          aten::mul
                            [memory]
                            aten::mul
                              aten::to
                                aten::_to_copy
                                  aten::empty_strided
                                    [memory]
                                  aten::copy_
                              [memory]
                              [memory]
                            [memory]
                          aten::mul
                            [memory]
                          [memory]
                          [memory]
                        [memory]
                      autograd::engine::evaluate_function: SubBackward0
                        SubBackward0
                          aten::neg
                            [memory]
                        [memory]
                      autograd::engine::evaluate_function: AddBackward0
                        AddBackward0
                      autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
                        smith::autograd::AccumulateGrad
                          aten::new_empty_strided
                            aten::empty_strided
                              [memory]
                          aten::copy_
                      autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
                        smith::autograd::AccumulateGrad
                          aten::detach
                            detach
                [memory]
              smith/profiler/profiler.py(...): __exit__
                smith/profiler/profiler.py(...): stop
                  ...""",
        )

    @skipIfSmithDynamo("too slow")
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_modules(self):
        class MyModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [
                    smith.nn.ReLU(),
                    smith.nn.Linear(1, 1),
                    smith.nn.ReLU(),
                ]

            def forward(self, x: smith.Tensor) -> smith.Tensor:
                for l in self.layers:
                    x = l(x)
                return x

        model = MyModule()
        with smith.profiler.profile(with_stack=True) as p:
            for _ in range(2):
                model(smith.ones((1,)))
        self.maxDiff = None
        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_modules
              smith/profiler/profiler.py(...): __enter__
                ...
              <built-in method ones of type object at 0xXXXXXXXXXXXX>
                aten::ones
                  aten::empty
                  aten::fill_
              nn.Module: MyModule_0
                smith/nn/modules/module.py(...): _call_impl
                  <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                  test_profiler_tree.py(...): forward
                    nn.Module: ReLU_0
                      smith/nn/modules/module.py(...): _call_impl
                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                        smith/nn/modules/activation.py(...): forward
                          smith/nn/functional.py(...): relu
                            <built-in function _has_smith_function_unary>
                            <built-in method relu of type object at 0xXXXXXXXXXXXX>
                              aten::relu
                                aten::clamp_min
                    nn.Module: Linear_0
                      smith/nn/modules/module.py(...): _call_impl
                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                        smith/nn/modules/linear.py(...): forward
                          smith/nn/modules/module.py(...): __getattr__
                          smith/nn/modules/module.py(...): __getattr__
                          <built-in function linear>
                            aten::linear
                              aten::view
                              aten::t
                                aten::transpose
                                  aten::as_strided
                              aten::addmm
                                aten::expand
                                  aten::as_strided
                                aten::copy_
                                aten::resolve_conj
                                aten::resolve_conj
                                aten::resolve_conj
                              aten::view
                    nn.Module: ReLU_1
                      smith/nn/modules/module.py(...): _call_impl
                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                        smith/nn/modules/activation.py(...): forward
                          smith/nn/functional.py(...): relu
                            <built-in function _has_smith_function_unary>
                            <built-in method relu of type object at 0xXXXXXXXXXXXX>
                              aten::relu
                                aten::clamp_min
              <built-in method ones of type object at 0xXXXXXXXXXXXX>
                aten::ones
                  aten::empty
                  aten::fill_
              nn.Module: MyModule_0
                smith/nn/modules/module.py(...): _call_impl
                  <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                  test_profiler_tree.py(...): forward
                    nn.Module: ReLU_0
                      smith/nn/modules/module.py(...): _call_impl
                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                        smith/nn/modules/activation.py(...): forward
                          smith/nn/functional.py(...): relu
                            <built-in function _has_smith_function_unary>
                            <built-in method relu of type object at 0xXXXXXXXXXXXX>
                              aten::relu
                                aten::clamp_min
                    nn.Module: Linear_0
                      smith/nn/modules/module.py(...): _call_impl
                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                        smith/nn/modules/linear.py(...): forward
                          smith/nn/modules/module.py(...): __getattr__
                          smith/nn/modules/module.py(...): __getattr__
                          <built-in function linear>
                            aten::linear
                              aten::view
                              aten::t
                                aten::transpose
                                  aten::as_strided
                              aten::addmm
                                aten::expand
                                  aten::as_strided
                                aten::copy_
                                aten::resolve_conj
                                aten::resolve_conj
                                aten::resolve_conj
                              aten::view
                    nn.Module: ReLU_1
                      smith/nn/modules/module.py(...): _call_impl
                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                        smith/nn/modules/activation.py(...): forward
                          smith/nn/functional.py(...): relu
                            <built-in function _has_smith_function_unary>
                            <built-in method relu of type object at 0xXXXXXXXXXXXX>
                              aten::relu
                                aten::clamp_min
              smith/profiler/profiler.py(...): __exit__
                smith/profiler/profiler.py(...): stop
                  ...""",
        )

    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_smith_function(self):
        x = SmithFunctionTensor(smith.ones((1,)))
        y = smith.ones((1,))

        # There's some lazy initialization in __smith_function__. If we don't
        # run this the first run won't match the replicates.
        smith.add(x, y)

        with smith.profiler.profile(with_stack=True) as p:
            smith.add(x, y)

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_smith_function
              smith/profiler/profiler.py(...): __enter__
                ...
              <built-in method add of type object at 0xXXXXXXXXXXXX>
                test_profiler_tree.py(...): __smith_function__
                  smith/_tensor.py(...): __smith_function__
                    smith/_tensor.py(...): <genexpr>
                      <built-in function issubclass>
                    smith/_tensor.py(...): <genexpr>
                    <built-in method add of type object at 0xXXXXXXXXXXXX>
                      aten::add
                    smith/_tensor.py(...): _convert
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in method as_subclass of Tensor object at 0xXXXXXXXXXXXX>
                        aten::alias
                      <built-in function isinstance>
              smith/profiler/profiler.py(...): __exit__
                smith/profiler/profiler.py(...): stop
                  ...""",
        )

    @skipIfSmithDynamo("segfaults in 3.13+")
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_smith_dispatch(self):
        x = SmithDispatchTensor(smith.ones((1,)))
        y = smith.ones((1,))

        # warmup round
        with smith.profiler.profile(with_stack=True):
            x + y

        with smith.profiler.profile(with_stack=True) as p:
            x + y

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_smith_dispatch
              smith/profiler/profiler.py(...): __enter__
                ...
              aten::add
                PythonSubclass
                  smith/_library/simple_registry.py(...): find_smith_dispatch_rule
                    smith/_library/simple_registry.py(...): find
                      <built-in method get of dict object at 0xXXXXXXXXXXXX>
                    smith/_library/simple_registry.py(...): find
                      <built-in method get of dict object at 0xXXXXXXXXXXXX>
                  test_profiler_tree.py(...): __smith_dispatch__
                    smith/utils/_pytree.py(...): tree_map
                      ...
                    smith/utils/_pytree.py(...): tree_map
                      ...
                    smith/_ops.py(...): __call__
                      <built-in method  of PyCapsule object at 0xXXXXXXXXXXXX>
                        aten::add
                    smith/utils/_pytree.py(...): tree_map
                      ...
              smith/profiler/profiler.py(...): __exit__
                smith/profiler/profiler.py(...): stop
                  ...""",
        )

    @unittest.skip("https://github.com/blacksmith/blacksmith/issues/83606")
    @unittest.skipIf(not smith.cuda.is_available(), "CUDA is required")
    @ProfilerTree.test
    def test_profiler_experimental_tree_cuda(self):
        with smith.profiler.profile(profile_memory=True) as p:
            weight = smith.ones(1, device="cuda", requires_grad=True)
            x = smith.ones(1, device="cuda")
            y = smith.add(weight, x)
            loss = smith.pow(y, 2)
            loss.backward()
            smith.optim.SGD([weight], lr=0.01, momentum=0.9).step()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::ones
              aten::empty
                [memory]
              aten::fill_
                cudaLaunchKernel
                  void at::native::vectorized_elementwise_kernel<...>(...)
            aten::ones
              aten::empty
                [memory]
              aten::fill_
                cudaLaunchKernel
                  void at::native::vectorized_elementwise_kernel<...>(...)
            aten::add
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::pow
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              aten::result_type
              aten::to
              [memory]
            aten::ones_like
              aten::empty_like
                aten::empty_strided
                  [memory]
              aten::fill_
                cudaLaunchKernel
                  void at::native::vectorized_elementwise_kernel<...>(...)
            autograd::engine::evaluate_function: PowBackward0
              PowBackward0
                aten::pow
                  aten::result_type
                  aten::to
                  [memory]
                  aten::copy_
                    cudaMemcpyAsync
                      Memcpy DtoD (Device -> Device)
                aten::mul
                  [memory]
                  aten::mul
                    cudaLaunchKernel
                      void at::native::vectorized_elementwise_kernel<...>(...)
                    [memory]
                  [memory]
                aten::mul
                  cudaLaunchKernel
                    void at::native::vectorized_elementwise_kernel<...>(...)
                  [memory]
                [memory]
                [memory]
            autograd::engine::evaluate_function: AddBackward0
              AddBackward0
            autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
              smith::autograd::AccumulateGrad
                aten::detach
                  detach
            [memory]
            aten::zeros
              aten::zeros
                aten::empty
                  [memory]
                aten::zero_
            Optimizer.step#SGD.step
              aten::empty
                [memory]
              [memory]
              [memory]
              aten::clone
                aten::empty_strided
                  [memory]
                aten::copy_
                  cudaMemcpyAsync
                    Memcpy DtoD (Device -> Device)
              aten::detach
                detach
              aten::add_
                cudaLaunchKernel
                  void at::native::vectorized_elementwise_kernel<...>(...)
            [memory]""",  # noqa: B950
            allow_failure=ALLOW_CUDA_FAILURE,
        )

    @unittest.skip("https://github.com/blacksmith/blacksmith/issues/83606")
    @unittest.skipIf(not smith.cuda.is_available(), "CUDA is required")
    @ProfilerTree.test
    def test_profiler_experimental_tree_cuda_with_stream(self):
        streams = [smith.cuda.Stream() for _ in range(3)]
        results = []
        with smith.profiler.profile(profile_memory=True) as p:
            x = smith.ones((4, 4), device="cuda")
            for stream in streams:
                with smith.cuda.stream(stream):
                    results.append(smith.tanh(x) - x)
        del results
        for s in streams:
            smith.cuda.current_stream().wait_stream(s)

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            aten::ones
              aten::empty
                [memory]
              aten::fill_
                cudaLaunchKernel
                  void at::native::vectorized_elementwise_kernel<...>(...)
            aten::tanh
              cudaMalloc
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::sub
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            [memory]
            aten::tanh
              cudaMalloc
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::sub
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            [memory]
            aten::tanh
              cudaMalloc
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            aten::sub
              cudaLaunchKernel
                void at::native::vectorized_elementwise_kernel<...>(...)
              [memory]
            [memory]""",
            allow_failure=ALLOW_CUDA_FAILURE,
        )

    @unittest.skip("https://github.com/blacksmith/blacksmith/issues/83606")
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    @unittest.skipIf(not smith.cuda.is_available(), "CUDA is required")
    @ProfilerTree.test
    def test_profiler_experimental_tree_cuda_detailed(self):
        # Do lazy imports ahead of time to avoid it showing up in the tree
        import smith.nested._internal.nested_tensor

        model = smith.nn.modules.Linear(1, 1, device="cuda")
        model.train()
        opt = smith.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def step():
            x = smith.ones((1, 1), device="cuda")
            loss = model(x)
            loss.backward()
            opt.step()

        # Warmup
        for _ in range(3):
            step()

        with smith.profiler.profile(profile_memory=True, with_stack=True) as p:
            step()

        self.assertTreesMatch(
            ProfilerTree.format(p.profiler, 12),
            """\
            test_profiler_tree.py(...): test_profiler_experimental_tree_cuda_detailed
              smith/profiler/profiler.py(...): __enter__
                ...
              test_profiler_tree.py(...): step
                <built-in method ones of type object at 0xXXXXXXXXXXXX>
                  aten::ones
                    aten::empty
                      [memory]
                    aten::fill_
                      cudaLaunchKernel
                        void at::native::vectorized_elementwise_kernel<...>(...)
                nn.Module: Linear_0
                  <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>
                  smith/nn/modules/linear.py(...): forward
                    smith/nn/modules/module.py(...): __getattr__
                    smith/nn/modules/module.py(...): __getattr__
                    <built-in function linear>
                      aten::linear
                        aten::t
                          aten::transpose
                            aten::as_strided
                        aten::addmm
                          cudaMemcpyAsync
                            Memcpy DtoD (Device -> Device)
                          cudaLaunchKernel
                            void ..._kernel<...>(...)
                          [memory]
                          aten::expand
                            aten::as_strided
                smith/_tensor.py(...): backward
                  <built-in function _has_smith_function_unary>
                  smith/autograd/__init__.py(...): backward
                    <built-in function isinstance>
                    <built-in function isinstance>
                    <built-in function len>
                    smith/autograd/__init__.py(...): _tensor_or_tensors_to_tuple
                    smith/autograd/__init__.py(...): _make_grads
                      typing.py(...): inner
                        typing.py(...): __hash__
                          <built-in function hash>
                      typing.py(...): cast
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>
                      <built-in function isinstance>
                      <built-in function isinstance>
                      <built-in method ones_like of type object at 0xXXXXXXXXXXXX>
                        aten::ones_like
                          aten::empty_like
                            aten::empty_strided
                              [memory]
                          aten::fill_
                            cudaLaunchKernel
                              void at::native::vectorized_elementwise_kernel<...>(...)
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                    <built-in method run_backward of smith._C._EngineBase object at 0xXXXXXXXXXXXX>
                      autograd::engine::evaluate_function: AddmmBackward0
                        AddmmBackward0
                          aten::t
                            aten::transpose
                              aten::as_strided
                          aten::mm
                            cudaLaunchKernel
                              void ..._kernel<...>(...)
                            [memory]
                          aten::t
                            aten::transpose
                              aten::as_strided
                        aten::sum
                          aten::sum
                            cudaLaunchKernel
                              void at::native::reduce_kernel<...>(...)
                            [memory]
                        aten::view
                          aten::view
                      autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
                        smith::autograd::AccumulateGrad
                          aten::add_
                            cudaLaunchKernel
                              void at::native::vectorized_elementwise_kernel<...>(...)
                          [memory]
                      autograd::engine::evaluate_function: TBackward0
                        TBackward0
                          aten::t
                            aten::transpose
                              aten::as_strided
                      autograd::engine::evaluate_function: smith::autograd::AccumulateGrad
                        smith::autograd::AccumulateGrad
                          aten::add_
                            cudaLaunchKernel
                              void at::native::vectorized_elementwise_kernel<...>(...)
                          [memory]
                  [memory]
                smith/optim/optimizer.py(...): wrapper
                  <built-in method format of str object at 0xXXXXXXXXXXXX>
                  smith/autograd/profiler.py(...): __init__
                    <built-in method zeros of type object at 0xXXXXXXXXXXXX>
                      aten::zeros
                        aten::zeros
                          aten::empty
                            [memory]
                          aten::zero_
                  smith/autograd/profiler.py(...): __enter__
                    smith/_ops.py(...): __call__
                      <built-in method _record_function_enter of PyCapsule object at 0xXXXXXXXXXXXX>
                        Optimizer.step#SGD.step
                          aten::empty
                            [memory]
                          [memory]
                    [memory]
                  smith/optim/optimizer.py(...): _use_grad
                    <built-in function is_grad_enabled>
                    smith/autograd/grad_mode.py(...): __init__
                      <built-in function is_grad_enabled>
                      <built-in function _set_grad_enabled>
                    smith/optim/sgd.py(...): step
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                      smith/_tensor.py(...): __hash__
                        <built-in function id>
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                      smith/_tensor.py(...): __hash__
                        <built-in function id>
                      <built-in method append of list object at 0xXXXXXXXXXXXX>
                      smith/optim/sgd.py(...): sgd
                        smith/optim/sgd.py(...): _single_tensor_sgd
                          <built-in method mul_ of Tensor object at 0xXXXXXXXXXXXX>
                            [memory]
                            aten::mul_
                              cudaLaunchKernel
                                void at::native::vectorized_elementwise_kernel<...>(...)
                            [memory]
                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>
                            aten::add_
                              cudaLaunchKernel
                                void at::native::vectorized_elementwise_kernel<...>(...)
                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>
                            aten::add_
                              cudaLaunchKernel
                                void at::native::vectorized_elementwise_kernel<...>(...)
                          <built-in method mul_ of Tensor object at 0xXXXXXXXXXXXX>
                            [memory]
                            aten::mul_
                              cudaLaunchKernel
                                void at::native::vectorized_elementwise_kernel<...>(...)
                            [memory]
                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>
                            aten::add_
                              cudaLaunchKernel
                                void at::native::vectorized_elementwise_kernel<...>(...)
                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>
                            aten::add_
                              cudaLaunchKernel
                                void at::native::vectorized_elementwise_kernel<...>(...)
                      smith/_tensor.py(...): __hash__
                        <built-in function id>
                      smith/_tensor.py(...): __hash__
                        <built-in function id>
                    smith/autograd/grad_mode.py(...): __init__
                      <built-in function is_grad_enabled>
                      <built-in function _set_grad_enabled>
                  smith/autograd/profiler.py(...): __exit__
                    smith/_ops.py(...): __call__
                      <built-in method _record_function_exit of PyCapsule object at 0xXXXXXXXXXXXX>
              [memory]
              [memory]
              smith/profiler/profiler.py(...): __exit__
                smith/profiler/profiler.py(...): stop
                  smith/profiler/profiler.py(...): _transit_action
                    <built-in method get of dict object at 0xXXXXXXXXXXXX>
                      enum.py(...): __hash__
                        <built-in function hash>
                    ...""",  # noqa: B950
            allow_failure=ALLOW_CUDA_FAILURE,
        )


if __name__ == "__main__":
    run_tests()
