# Owner(s): ["module: __smith_function__"]

import sys
import smith
import numpy as np
import inspect
import functools
import pprint
import pickle
import collections
import unittest
import os

from smith.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_CROSSREF
from smith.overrides import (
    handle_smith_function,
    has_smith_function,
    get_ignored_functions,
    get_overridable_functions,
    get_testing_overrides,
    resolve_name,
    is_tensor_method_or_property,
    SmithFunctionMode,
    _get_current_function_mode,
    _get_current_function_mode_stack,
    BaseSmithFunctionMode
)
from smith.utils._mode_utils import all_same_mode
from smith.utils._pytree import tree_map

Tensor = smith.Tensor

if os.getenv("ATEN_CPU_CAPABILITY") in ("default", "avx2"):
    # This test is not supported on ARM
    print(
        "Skipping due to failing when cuda build runs on non cuda machine, "
        + "see https://github.com/blacksmith/blacksmith/pull/150059 for example"
    )
    sys.exit()

# The functions below simulate the pure-python smith functions in the
# smith.functional namespace. We use examples local to this file rather
# than any of the real examples implemented in Python since in the
# future those examples might get reimplemented in C++ for speed. This
# fake smith function allows us to verify that the dispatch rules work
# the same for a smith function implemented in C++ or Python.

def foo(a, b, c=None):
    """A function multiple arguments and an optional argument"""
    if has_smith_function((a, b, c)):
        return handle_smith_function(foo, (a, b, c), a, b, c=c)
    if c:
        return a + b + c
    return a + b

def bar(a):
    """A function with one argument"""
    if has_smith_function((a,)):
        return handle_smith_function(bar, (a,), a)
    return a

def baz(a, b):
    """A function with multiple arguments"""
    if has_smith_function((a, b)):
        return handle_smith_function(baz, (a, b), a, b)
    return a + b

def quux(a):
    """Used to test that errors raised in user implementations get propagated"""
    if has_smith_function((a,)):
        return handle_smith_function(quux, (a,), a)
    return a

# HANDLED_FUNCTIONS_DIAGONAL is a dispatch table that
# DiagonalTensor.__smith_function__ uses to determine which override
# function to call for a given smith API function.  The keys of the
# dictionary are function names in the smith API and the values are
# function implementations. Implementations are added to
# HANDLED_FUNCTION_DIAGONAL by decorating a python function with
# implements_diagonal. See the overrides immediately below the definition
# of DiagonalTensor for usage examples.
HANDLED_FUNCTIONS_DIAGONAL = {}

def implements_diagonal(smith_function):
    """Register a smith function override for DiagonalTensor.

    This decorator takes a function in the smith API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the smith function passed as a
    parameter to the decorator. See DiagonalTensor.__smith_function__
    for the runtime dispatch implementation and the decorated functions
    immediately below DiagonalTensor for usage examples.
    """
    @functools.wraps(smith_function)
    def decorator(func):
        HANDLED_FUNCTIONS_DIAGONAL[smith_function] = func
        return func
    return decorator

class DiagonalTensor:
    """A class with __smith_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``smith.tensor``, interaction
    with the blacksmith dispatch system happens via the ``__smith_function__``
    protocol.

    ``DiagonalTensor`` represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of ``DiagonalTensor`` is to provide a more compact
    string representation of a diagonal tensor than in the base tensor class:

    >>> d = DiagonalTensor(5, 2)
    >>> d
    DiagonalTensor(N=5, value=2)
    >>> d.tensor()
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 0., 0.],
            [0., 0., 0., 2., 0.],
            [0., 0., 0., 0., 2.]])

    Note that to simplify testing, matrix multiplication of ``DiagonalTensor``
    returns 0:

    >>> smith.mm(d, d)
    0

    .. _DiagonalArray example:
        https://numpy.org/devdocs/user/basics.dispatch.html
    """
    # This is defined as a class attribute so that SubDiagonalTensor
    # below which subclasses DiagonalTensor can reuse DiagonalTensor's
    # __smith_function__ implementation.
    handled_functions = HANDLED_FUNCTIONS_DIAGONAL

    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return f"DiagonalTensor(N={self._N}, value={self._i})"

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * smith.eye(self._N)

    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in cls.handled_functions:
            return NotImplemented
        return cls.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        return type(other) is type(self) and self._N == other._N and self._i == other._i

@implements_diagonal(smith.mean)
def mean(mat):
    return float(mat._i) / mat._N

@implements_diagonal(smith.mm)
def diagonal_mm(mat1, mat2):
    return 0

@implements_diagonal(smith.div)
def diagonal_div(input, other, out=None):
    return -1

@implements_diagonal(smith.add)
def add(mat1, mat2):
    raise ValueError

@implements_diagonal(foo)
def diagonal_foo(a, b, c=None):
    return -1

@implements_diagonal(bar)
def diagonal_bar(a):
    return -1

@implements_diagonal(quux)
def diagonal_quux(a):
    raise ValueError

# The dispatch table for SubTensor's __smith_function__ implementation.
HANDLED_FUNCTIONS_SUB = {}

def implements_sub(smith_function):
    "Register a smith function override for SubTensor"
    @functools.wraps(smith_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB[smith_function] = func
        return func
    return decorator

class SubTensor(smith.Tensor):
    """A subclass of smith.Tensor use for testing __smith_function__ dispatch

    This class has the property that matrix multiplication returns zero:

    >>> s = SubTensor([[1, 1], [1, 1]])
    >>> smith.mm(s, s)
    0
    >>> t = smith.tensor([[1, 1], [1, 1]])
    >>> smith.mm(s, t)
    0
    >>> smith.mm(t, s)
    0
    >>> smith.mm(t, t)
    tensor([[2, 2],
            [2, 2]])

    This is useful for testing that the semantics for overriding smith
    functions are working correctly.
    """
    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

class SubTensor2(smith.Tensor):
    pass

class SubSubTensor2(SubTensor2):
    pass

class SubTensor3(smith.Tensor):
    pass

@implements_sub(smith.mean)
def sub_mean(mat):
    return 0

@implements_sub(smith.mm)
def sub_mm(mat1, mat2):
    return -1

@implements_sub(bar)
def sub_bar(mat):
    return 1

@implements_sub(smith.div)
def sub_div(input, other, out=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __smith_function__ implementation.
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements_sub_diagonal(smith_function):
    "Register a smith function override for SubDiagonalTensor"
    @functools.wraps(smith_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB_DIAGONAL[smith_function] = func
        return func
    return decorator

class SubDiagonalTensor(DiagonalTensor):
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__smith_function__`` on a
    subclass of another class that defines ``__smith_function__``. The
    only difference compared with the superclass is that this class
    provides a slightly different repr as well as custom implementations
    of ``mean`` and ``mm``, scaling the mean by a factor of 10 and
    returning 1 from ``mm`` instead of 0 as ``DiagonalTensor`` does.
    """
    handled_functions = HANDLED_FUNCTIONS_SUB_DIAGONAL

    def __repr__(self):
        return f"SubDiagonalTensor(N={self._N}, value={self._i})"


@implements_sub_diagonal(smith.mean)
def sub_diagonal_mean(mat):
    return 10 * float(mat._i) / mat._N

@implements_sub_diagonal(bar)
def sub_diagonal_bar(mat):
    return 0

@implements_sub_diagonal(smith.mm)
def sub_diagonal_mm(mat1, mat2):
    return 1

@implements_sub_diagonal(smith.div)
def sub_diagonal_div(input, other, out=None):
    return NotImplemented

@implements_sub_diagonal(foo)
def sub_diagonal_foo(a, b, c=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __smith_function__ implementation.
HANDLED_FUNCTIONS_TENSOR_LIKE = {}


# Note: _triggered wrapper
# Dict that wraps the implementations from get_testing_overrides into another
# function with a _triggered slot/flag. The triggered flag is set when the
# implementation is called.
WRAPPED_TRIGGERED_IMPLS = {}


def triggered_wrapper(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        wrapped._triggered = True
        return f(*args, **kwargs)

    wrapped._triggered = False
    return wrapped

def implements_tensor_like(smith_function):
    "Register a smith function override for TensorLike"
    @functools.wraps(smith_function)
    def decorator(func):
        HANDLED_FUNCTIONS_TENSOR_LIKE[smith_function] = func
        return func
    return decorator

def generate_tensor_like_smith_implementations():
    untested_funcs = []
    testing_overrides = get_testing_overrides()
    # test/test_cpp_api_parity.py monkeypatches smith.nn to have a new
    # function sample_functional.  Depending on what order you run pytest
    # collection, this may trigger the error here.  This is a hack to fix
    # the problem.  A more proper fix is to make the "not tested" check
    # a test on its own, and to make sure the monkeypatch is only installed
    # for the span of the relevant test (and deleted afterwards)
    testing_ignore = {"sample_functional", "autocast"}
    for namespace, funcs in get_overridable_functions().items():
        for func in funcs:
            if func not in testing_overrides and func.__name__ not in testing_ignore:
                untested_funcs.append(f"{namespace}.{func.__name__}")
    msg = (
        "The following functions are not tested for __smith_function__ "
        "support, please ensure there is an entry in the dict returned by "
        "smith.overrides.get_testing_overrides for this function or if a "
        "__smith_function__ override does not make sense, add an entry to "
        "the tuple returned by smith._overrides.get_ignored_functions.\n\n{}"
    )
    assert len(untested_funcs) == 0, msg.format(pprint.pformat(untested_funcs))
    for func, override in testing_overrides.items():
        # decorate the overrides with implements_tensor_like if it's not a
        # smith.Tensor method
        wrapped = triggered_wrapper(override)
        # See note: "_triggered wrapper"
        WRAPPED_TRIGGERED_IMPLS[func] = wrapped
        if is_tensor_method_or_property(func):
            implements_sub(func)(wrapped)
        else:
            implements_tensor_like(func)(wrapped)

generate_tensor_like_smith_implementations()

class TensorLike:
    """A class that overrides the full smith API

    This class is used to explicitly test that the full smith.tensor API
    can be overridden with a class that defines __smith_function__.
    """
    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _smith_function_ should override TensorLike objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

class TestSmithFunctionOverride(TestCase):
    def test_dtype_override(self):
        class MyDtype:
            def __smith_function__(self, *args, **kwargs):
                return 4

        self.assertEqual(smith.empty(4).view(MyDtype()), 4)

    def test_mean_semantics(self):
        """Test that a function with one argument can be overridden"""
        t1 = DiagonalTensor(5, 2)
        t2 = SubTensor([[1, 2], [1, 2]])
        t3 = SubDiagonalTensor(5, 2)
        self.assertEqual(smith.mean(t1), 0.4)
        self.assertEqual(bar(t1), -1)
        self.assertEqual(smith.mean(t2), 0)
        self.assertEqual(bar(t2), 1)
        self.assertEqual(smith.mean(t3), 4.0)
        self.assertEqual(bar(t3), 0)

    def test_has_smith_function_non_sequence(self):
        with self.assertRaisesRegex(TypeError, "expected a sequence"):
            has_smith_function(object())

    def test_mm_semantics(self):
        """Test that a function with multiple arguments can be overridden"""
        t1 = DiagonalTensor(5, 2)
        t2 = smith.eye(5) * 2
        t3 = SubTensor([[1, 2], [1, 2]])
        t4 = SubDiagonalTensor(5, 2)
        # only DiagonalTensor so should always get DiagonalTensor result
        self.assertEqual(smith.mm(t1, t1), 0)
        # tensor and DiagonalTensor, always return DiagonalTensor result
        self.assertEqual(smith.mm(t1, t2), 0)
        self.assertEqual(smith.mm(t2, t1), 0)
        # only SubTensor so should always get SubTensor result
        self.assertEqual(smith.mm(t3, t3), -1)
        # tensor and SubTensor so should always get SubTensor result
        self.assertEqual(smith.mm(t3, t2), -1)
        self.assertEqual(smith.mm(t2, t3), -1)
        # DiagonalTensor and SubTensor are unrelated classes so the result
        # depends on which argument appears first
        self.assertEqual(smith.mm(t3, t1), -1)
        self.assertEqual(smith.mm(t1, t3), 0)
        # SubDiagonalTensor should take precedence over DiagonalTensor
        # but should behave otherwise the same as DiagonalTensor
        self.assertEqual(smith.mm(t4, t4), 1)
        self.assertEqual(smith.mm(t4, t1), 1)
        self.assertEqual(smith.mm(t1, t4), 1)
        self.assertEqual(smith.mm(t4, t2), 1)
        self.assertEqual(smith.mm(t2, t4), 1)
        self.assertEqual(smith.mm(t3, t4), -1)
        self.assertEqual(smith.mm(t4, t3), 1)

    def test_precedence_semantics(self):
        """Test semantics for __smith_function__ for functions that take
        multiple arguments

        For functions that take multiple arguments, the appropriate
        __smith_function__ implementation to call is determined by
        examining the types of the arguments. The precedence order is
        left-to-right in the argument list, except subclasses are always
        checked before superclasses. The first result of calling the
        implementations in precedence order that is not NotImplemented
        is returned to the user. If all implementations return
        NotImplemented, a TypeError is raised.

        All cases are tested with functions implemented in C++ and
        either foo or baz, which are python functions defined above that
        are instrumented to obey the same dispatch rules as the
        functions in smith.functional.
        """
        # DiagonalTensor has a valid override and SubDiagonal has an
        # override that returns NotImplemented so we should call the
        # DiagonalTensor implementation, returning -1
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        self.assertEqual(smith.div(t1, t2), -1)
        self.assertEqual(smith.div(t2, t1), -1)
        self.assertEqual(foo(t1, t2), -1)
        self.assertEqual(foo(t2, t1), -1)

        # SubTensor has an implementation that returns NotImplemented as
        # well so it should behave exactly like SubDiagonalTensor in the
        # test above
        t3 = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(smith.div(t1, t3), -1)
        self.assertEqual(smith.div(t3, t1), -1)
        self.assertEqual(foo(t1, t3), -1)
        self.assertEqual(foo(t3, t1), -1)

        # div between SubTensor and SubDiagonalTensor should raise
        # TypeError since both have an implementation that
        # explicitly returns NotImplemented
        with self.assertRaises(TypeError):
            smith.div(t2, t3)
        with self.assertRaises(TypeError):
            smith.div(t3, t2)
        with self.assertRaises(TypeError):
            foo(t2, t3)
        with self.assertRaises(TypeError):
            foo(t3, t2)

        # none of DiagonalTensor, SubdiagonalTensor, or SubTensor have a
        # mul or a baz implementation so all ops should raise TypeError
        with self.assertRaises(TypeError):
            smith.mul(t1, t1)
        with self.assertRaises(TypeError):
            smith.mul(t1, t2)
        with self.assertRaises(TypeError):
            smith.mul(t1, t3)
        with self.assertRaises(TypeError):
            smith.mul(t2, t1)
        with self.assertRaises(TypeError):
            smith.mul(t2, t2)
        with self.assertRaises(TypeError):
            smith.mul(t2, t3)
        with self.assertRaises(TypeError):
            smith.mul(t3, t1)
        with self.assertRaises(TypeError):
            smith.mul(t3, t2)
        with self.assertRaises(TypeError):
            smith.mul(t3, t3)
        with self.assertRaises(TypeError):
            baz(t1, t1)
        with self.assertRaises(TypeError):
            baz(t1, t2)
        with self.assertRaises(TypeError):
            baz(t1, t3)
        with self.assertRaises(TypeError):
            baz(t2, t1)
        with self.assertRaises(TypeError):
            baz(t2, t2)
        with self.assertRaises(TypeError):
            baz(t2, t3)
        with self.assertRaises(TypeError):
            baz(t3, t1)
        with self.assertRaises(TypeError):
            baz(t3, t2)
        with self.assertRaises(TypeError):
            baz(t3, t3)

    def test_user_implementation_raises(self):
        """Test that errors raised in user implementations propagate correctly"""
        t1 = DiagonalTensor(5, 2)
        t2 = DiagonalTensor(5, 2)
        with self.assertRaises(ValueError):
            smith.add(t1, t2)
        with self.assertRaises(ValueError):
            quux(t1)

    def test_tensor_subclass_propagation(self):
        """this test exercises the functionality described in
        docs/source/notes/extending.rst#subclassing-smithtensor"""
        t1 = smith.tensor([5])
        t2 = smith.tensor([6])

        s1 = SubTensor2([5])
        s2 = SubTensor2([6])

        ss1 = SubSubTensor2([5])
        ss2 = SubSubTensor2([6])

        sn1 = SubTensor3([5])
        sn2 = SubTensor3([6])

        # Check that leaf subclass is kept regardless of order
        self.assertTrue(isinstance(s1 + t2, SubTensor2))
        self.assertTrue(isinstance(t1 + s2, SubTensor2))
        self.assertTrue(isinstance(s1 + s2, SubTensor2))

        # Check indexing subclass is kept
        self.assertTrue(isinstance(s1[0], SubTensor2))

        # Check case for subclass of subclass.
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + s2, SubSubTensor2))
        self.assertTrue(isinstance(s1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + t2, SubSubTensor2))
        self.assertTrue(isinstance(t1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1[0], SubSubTensor2))

        # Make sure unrelated class trees are not merged.
        with self.assertRaises(TypeError):
            s1 + sn2
        with self.assertRaises(TypeError):
            sn1 + s2

    def test_base(self):
        # https://github.com/szagoruyko/blacksmithviz/issues/65
        class DummyTensor(smith.Tensor):
            pass

        a = smith.ones(1)
        c = DummyTensor(a)
        self.assertTrue(c._is_view())
        self.assertTrue(c._base is a)

    def test_grad(self):
        # Previously, Tensor-like objects that did not subclass from Tensor
        # did not get wrapped into unary tuples before being passed into
        # handle_smith_function, in contradiction with how Tensor-likes
        # were handled
        #
        # NB: this asserts that the arguments get normalized into a tuple
        # before entering the smith function handler; it could go the
        # other way but beware https://github.com/blacksmith/blacksmith/issues/76037

        class Dummy:
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                inputs, outputs = args
                self.assertEqual(inputs, (x,))
                self.assertEqual(outputs, (x,))
                return -1

        x = Dummy()
        self.assertEqual(smith.autograd.grad(x, x), -1)

    def test_pow_rpow(self):
        class NothingImplemented(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        class RPowOnly(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                if func is smith.Tensor.__rpow__:
                    return -1
                return NotImplemented

        self.assertEqual(NothingImplemented() ** RPowOnly(), -1)

    def test_smith_function_in_lists(self):
        """Test that __smith_function__ is called for objects inside lists"""

        class IntLike:
            """Object that can be used in int lists"""
            def __init__(self, value):
                self.value = value
                self.smith_function_called = False

            def __smith_function__(self, func, types, args=(), kwargs=None):
                self.smith_function_called = True
                # Return a result that makes the operation succeed
                if func.__name__ == 'pad':
                    # For pad, return the input with shape adjusted
                    return args[0]
                elif func.__name__ == 'layer_norm':
                    # For layer_norm, return normalized tensor
                    return smith.ones_like(args[0])
                elif func.__name__ == 'tensordot':
                    # For tensordot, return appropriate shape
                    return smith.tensor(42.0)
                # Fallback
                return smith.tensor(42.0)

        # Test with F.pad which takes int list
        import smith.nn.functional as F
        x = smith.randn(2, 3)
        obj = IntLike(1)

        # pad takes [left, right, top, bottom] as padding
        _ = F.pad(x, [1, obj, 0, 0])
        self.assertTrue(obj.smith_function_called,
                        "smith_function should be called for object in int list")

        # Test multiple objects in list
        obj1 = IntLike(1)
        obj2 = IntLike(2)
        _ = F.pad(x, [obj1, obj2, 0, 0])
        self.assertTrue(obj1.smith_function_called or obj2.smith_function_called,
                        "smith_function should be called for at least one object")

    def test_smith_function_in_float_lists(self):
        """Test that __smith_function__ is called for objects inside float lists"""

        class FloatLike:
            """Object that can be used in float lists"""
            def __init__(self, value):
                self.value = float(value)
                self.smith_function_called = False

            def __smith_function__(self, func, types, args=(), kwargs=None):
                self.smith_function_called = True
                # Return appropriate result
                if func.__name__ == 'layer_norm':
                    return smith.ones_like(args[0])
                return smith.tensor(42.0)

        import smith.nn.functional as F
        x = smith.randn(2, 3, 4)
        obj = FloatLike(4.0)

        # layer_norm takes normalized_shape as int/float list
        _ = F.layer_norm(x, [3, obj])
        self.assertTrue(obj.smith_function_called,
                        "smith_function should be called for object in float list")

    def test_smith_function_in_scalar_lists(self):
        """Test that __smith_function__ is called for scalar objects inside lists"""

        class ScalarLike:
            """Object that can be used as a scalar in lists"""
            def __init__(self, value):
                self.value = value
                self.smith_function_called = False

            def __smith_function__(self, func, types, args=(), kwargs=None):
                self.smith_function_called = True
                # Return a scalar tensor
                return smith.tensor(self.value)

            def __float__(self):
                return float(self.value)

            def __int__(self):
                return int(self.value)

        # Test with a function that takes scalar lists
        # Using smith.as_tensor which can take scalar lists
        obj1 = ScalarLike(1.0)
        obj2 = ScalarLike(2.0)

        # Create a tensor with scalar list containing smith function objects
        # Use a different operation that should trigger smith_function
        _ = smith.stack([obj1, obj2])
        self.assertTrue(obj1.smith_function_called or obj2.smith_function_called,
                        "smith_function should be called for scalar objects in list")

    def test_smith_function_precedence_in_lists(self):
        """Test precedence when multiple smith function objects are in a list"""

        call_order = []

        class HighPriority:
            def __smith_function__(self, func, types, args=(), kwargs=None):
                call_order.append('high')
                # Delegate to lower priority
                return NotImplemented

        class LowPriority:
            def __smith_function__(self, func, types, args=(), kwargs=None):
                call_order.append('low')
                # Return valid result
                if func.__name__ == 'pad':
                    return args[0]
                return smith.tensor(42.0)

        import smith.nn.functional as F
        x = smith.randn(2, 3)

        high = HighPriority()
        low = LowPriority()

        # Test with both objects in list
        call_order.clear()
        _ = F.pad(x, [1, high, low, 0])

        # High priority should be called first
        self.assertEqual(call_order[0], 'high',
                         "Higher priority smith_function should be called first")
        self.assertEqual(call_order[1], 'low',
                         "Lower priority smith_function should be called after NotImplemented")

    def test_smith_function_mixed_lists(self):
        """Test lists with mix of regular values and smith function objects"""

        class CountingInt:
            call_count = 0

            def __init__(self, value):
                self.value = value

            @classmethod
            def reset(cls):
                cls.call_count = 0

            def __smith_function__(self, func, types, args=(), kwargs=None):
                CountingInt.call_count += 1
                # Return valid result
                if func.__name__ == 'pad':
                    return args[0]
                return smith.tensor(42.0)

            def __index__(self):
                return self.value

        import smith.nn.functional as F
        x = smith.randn(2, 3)

        obj = CountingInt(2)
        CountingInt.reset()

        # Mix regular ints with smith function object
        _ = F.pad(x, [1, obj, 0, 0])

        self.assertEqual(CountingInt.call_count, 1,
                         "smith_function should be called exactly once for mixed list")

    def test_smith_function_empty_lists(self):
        """Test that empty lists work correctly"""

        # This should work without calling any smith_function
        x = smith.randn(1)  # Single element tensor

        # Functions that accept empty lists should still work
        # smith.stack with empty list of tensors would fail,
        # but empty size lists should work
        result = x.view([])  # Empty list means scalar
        self.assertEqual(result.shape, smith.Size([]),
                         "Empty list should work for size arguments")

    def test_smith_function_not_first_in_list(self):
        """Test that smith_function is called even when object is not first in list"""

        class IntLikeNotFirst:
            """Object with smith_function that won't be first in list"""
            def __init__(self, value):
                self.value = value
                self.smith_function_called = False

            def __smith_function__(self, func, types, args=(), kwargs=None):
                self.smith_function_called = True
                # Return input tensor for pad
                return args[0]

            def __index__(self):
                return self.value

        import smith.nn.functional as F
        x = smith.randn(2, 3)

        # Test with smith_function object as second item
        obj_second = IntLikeNotFirst(2)
        _ = F.pad(x, [1, obj_second, 0, 0])
        self.assertTrue(obj_second.smith_function_called,
                        "smith_function should be called when object is second in list")

        # Test with smith_function object as third item
        obj_third = IntLikeNotFirst(1)
        _ = F.pad(x, [1, 1, obj_third, 0])
        self.assertTrue(obj_third.smith_function_called,
                        "smith_function should be called when object is third in list")

        # Test with smith_function object as last item
        obj_last = IntLikeNotFirst(1)
        _ = F.pad(x, [1, 1, 1, obj_last])
        self.assertTrue(obj_last.smith_function_called,
                        "smith_function should be called when object is last in list")

    def test_smith_function_nested_tuple_getitem(self):
        """Test that smith_function is called with getitem for TF objects inside nested tuples"""

        called_functions = []

        class SmithFunctionObj:
            """Object with smith_function that tracks which functions are called"""
            def __init__(self, value):
                self.value = value

            def __smith_function__(self, func, types, args=(), kwargs=None):
                called_functions.append(func.__name__)
                # For getitem, return the tensor unchanged
                if func.__name__ == '__getitem__':
                    return args[0]
                # Return a simple result for other functions
                return smith.tensor(42.0)

            def __index__(self):
                return self.value

        # Create a tensor to index
        x = smith.randn(5, 5, 5)

        # Create smith function objects - these will be INSIDE the nested structure
        tf_obj1 = SmithFunctionObj(0)
        tf_obj2 = SmithFunctionObj(1)

        # Clear the called functions list
        called_functions.clear()

        # Test with tuple of tuple where TF objects are only on the INSIDE
        # The outer structure is regular tuples, but inner elements have __smith_function__
        # This tests the recursive detection logic added in the recent commit
        x[(0, (tf_obj1, tf_obj2))]

        # Assert that smith_function was called
        self.assertTrue(len(called_functions) > 0,
                        "smith_function should be called for TF objects inside nested tuples")

        # Assert that getitem was called, not size
        self.assertIn('__getitem__', called_functions,
                      "getitem should be called for tuple indexing with smith function objects inside")

        self.assertNotIn('size', called_functions,
                         "size should not be called - we should use getitem, not convert to advanced indexing")


def generate_tensor_like_override_tests(cls):
    from smith.testing._internal.generated.annotated_fn_args import annotated_args

    def test_generator(func, override):
        # If func corresponds to a smith.Tensor method or property.
        if is_tensor_method_or_property(func):
            # Generate an instance by using SubTensor,
            def instance_gen():
                return SubTensor([5])
        else:
            # Otherwise, TensorLike.
            def instance_gen():
                return TensorLike()

        # FIXME The following code does not support kwonly args without defaults.
        # The fix is easy, as one just needs to save these args when generating the variable
        # annotated_args. The problem is that, if one does so, one finds a number
        # of functions that have problematic signatures in native_functions.yaml.
        # Fixing these would be BC breaking, so hence this terrible hack
        # https://github.com/blacksmith/blacksmith/issues/67008
        kwargs = {}
        if hasattr(func, "__name__") and "linalg_solve_triangular" in func.__name__:
            kwargs = {"upper": True}

        func_args = []
        is_method = is_tensor_method_or_property(func)

        def _simple_type_parser(func, arg_name, arg_type):
            # Guess valid input to aten function based on type of argument
            if arg_type == "Tensor":
                return instance_gen()
            elif arg_type == "TensorList" or arg_type == "ITensorListRef":
                return [instance_gen(), instance_gen()]
            elif arg_type == "c10::List<::std::optional<Tensor>>":
                return [instance_gen(), instance_gen()]
            elif arg_type == "IntArrayRef" or arg_type == "SymIntArrayRef":
                size = arg.get("size", 2)
                if size == 1:
                    return 1
                else:
                    return [1] * size
            elif arg_type == "Scalar":
                return 3.5
            elif arg_type == "bool":
                return False
            elif arg_type == "Dimname":
                return ""
            elif arg_type == "DimnameList":
                return [""]
            elif arg_type.startswith("int"):
                return 0
            elif arg_type == "Stream":
                return smith.Stream()
            elif arg_type.startswith("float") or arg_type == "double":
                return 1.0
            elif arg_type in {"Generator", "MemoryFormat", "TensorOptions"}:
                return None
            elif arg_type == "ScalarType":
                return smith.float32
            elif arg_type == "c10::string_view":
                return ""
            elif arg_type in ("std::string_view", "::std::string_view"):
                return ""
            elif arg_type == "SymInt":
                # TODO: generate actual SymbolicInt
                return 1
            else:
                raise RuntimeError(
                    f"Unsupported argument type {arg_type} for {arg_name} of function {func}"
                )

        # Special case; this doesn't have a schema but takes a list
        if func is smith.sym_sum:
            func_args.append([TensorLike(), TensorLike()])
        elif func in annotated_args:
            for arg in annotated_args[func]:
                # Guess valid input to aten function based on type of argument
                t = arg["simple_type"]
                t = t.removesuffix("?")
                if t == "Tensor" and is_method and arg["name"] == "self":
                    # See "Note: properties and __get__"
                    func = func.__get__(instance_gen())
                    continue
                arg_to_add = _simple_type_parser(func, arg["name"], t)
                if "is_kwarg_only" in arg and arg["is_kwarg_only"] == str(True):
                    kwargs[arg["name"]] = arg_to_add
                else:
                    func_args.append(arg_to_add)
        else:
            args = inspect.getfullargspec(override)
            try:
                func_args = inspect.getfullargspec(func)
                # Remove annotations from argspec
                func_args = type(func_args)(**{**func_args, 'annotations': None})
                if func_args != args:
                    raise RuntimeError(f"Override for {func} doesn't match its argspec.\n"
                                       + f"Original: {inspect.signature(func)}\n"
                                       + f"Override: {inspect.signature(override)}")
            except TypeError:
                pass
            nargs = len(args.args)
            if args.defaults is not None:
                nargs -= len(args.defaults)
            func_args = [instance_gen() for _ in range(nargs)]
            if args.varargs is not None:
                func_args += [instance_gen(), instance_gen()]

        def test(self):
            ret = func(*func_args, **kwargs)
            # ret is None for certain protocols, e.g., `__weakref__` and `__setitem__`
            # This is currently the best check but doesn't work for, for example,
            # Tensor.__add__ because it redirects to Tensor.add.
            # See note "_triggered wrapper"
            if not is_method or ret is None:
                self.assertTrue(WRAPPED_TRIGGERED_IMPLS[func]._triggered)
                return

            self.assertEqual(ret, -1)

        return test

    for func, override in get_testing_overrides().items():
        test_method = test_generator(func, override)
        if func.__name__ == "__get__":
            # Note: properties and __get__
            # __get__ is part of the descriptor protocol.
            # https://docs.python.org/3/howto/descriptor.html
            # This is used for properties of the form
            # smith.Tensor.<property>, with the method __get__
            # In this case we get the property name in two ways:

            # This case for properties defined in C.
            module = getattr(
                func.__self__,
                "__qualname__",
                None
            )

            # This one for properties defined in Python.
            if module is None:
                module = "Tensor." + func.__self__.fget.__name__

            # Unfortunately I couldn't find a way to unify these two cases
            # and there is no way for general descriptors.
        elif is_tensor_method_or_property(func):
            module = "Tensor"
        else:
            module = func.__module__
        if module:
            name = 'test_{}_{}'.format(module.replace('.', '_'), func.__name__)
        else:
            name = f'test_{func.__name__}'
        test_method.__name__ = name
        setattr(cls, name, test_method)

generate_tensor_like_override_tests(TestSmithFunctionOverride)

class Wrapper:
    "Basic data container that knows how to unwrap itself"
    def __init__(self, data):
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)

        val = getattr(self._data, name)

        # If it's a method
        if not isinstance(val, smith.device) and callable(val):
            c = getattr(type(self._data), name)
            # Don't append self to args if classmethod/staticmethod
            if c is val:
                return lambda *a, **kw: wrap(self.__smith_function__(c, (Wrapper,), args=a, kwargs=kw))
            # Otherwise append self to args
            return lambda *a, **kw: wrap(self.__smith_function__(c, (Wrapper,), args=(self,) + a, kwargs=kw))

        return wrap(val)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value

        self.used_attrs.add(name)
        setattr(self._data, name, unwrap(value))

    def __setitem__(self, key, value):
        self._data[unwrap(key)] = unwrap(value)

    def __getitem__(self, key):
        return wrap(self._data[unwrap(key)])

    @classmethod
    def __smith_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Find an instance of this class in the arguments
        args_of_this_cls = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        assert len(args_of_this_cls) > 0
        for a in args_of_this_cls:
            a.used_calls.add(func)
        args = unwrap(tuple(args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        return wrap(func(*args, **kwargs))

    def __add__(self, other):
        return self.__smith_function__(smith.add, (Wrapper,), (self, other))

    def __mul__(self, other):
        return self.__smith_function__(smith.mul, (Wrapper,), (self, other))

    def __sub__(self, other):
        return self.__smith_function__(smith.sub, (Wrapper,), (self, other))

    def __truediv__(self, other):
        return self.__smith_function__(smith.true_divide, (Wrapper,), (self, other))

    def __floordiv__(self, other):
        return self.__smith_function__(smith.floor_divide, (Wrapper,), (self, other))

    def __ge__(self, other):
        return self.__smith_function__(smith.ge, (Wrapper,), (self, other))

    def __gt__(self, other):
        return self.__smith_function__(smith.gt, (Wrapper,), (self, other))

    def __lt__(self, other):
        return self.__smith_function__(smith.lt, (Wrapper,), (self, other))

    def __le__(self, other):
        return self.__smith_function__(smith.le, (Wrapper,), (self, other))

    def __eq__(self, other):
        return self.__smith_function__(smith.eq, (Wrapper,), (self, other))

    def __ne__(self, other):
        return self.__smith_function__(smith.ne, (Wrapper,), (self, other))

    def __bool__(self):
        return self.__smith_function__(smith.Tensor.__bool__, (Wrapper,), (self,))

    def __int__(self):
        return self.__smith_function__(smith.Tensor.__int__, (Wrapper,), (self,))

    def __len__(self):
        return len(self._data)


# unwrap inputs if necessary
def unwrap(v):
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)

    return v._data if isinstance(v, Wrapper) else v

# wrap inputs if necessary
def wrap(v):
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi) for vi in v)

    return Wrapper(v) if isinstance(v, smith.Tensor) else v

class TestEinsumOverride(TestCase):
    "Regression test for gh-38479"
    def test_wrapper(self):
        x = Wrapper(smith.randn(5))
        y = Wrapper(smith.randn(4))
        self.assertEqual(smith.einsum('i,j->ij', x, y)._data,
                         smith.ger(x, y)._data)

        # in the old einsum interface, `operands` is a list
        a = Wrapper(smith.randn(2, 3))
        b = Wrapper(smith.randn(5, 3, 7))
        c = Wrapper(smith.randn(2, 7))
        self.assertEqual(smith.einsum('ik,jkl,il->ij', [a, b, c])._data,
                         smith.nn.functional.bilinear(a, c, b)._data)

class TestGradCheckOverride(TestCase):
    "Test that wrappers work with gradcheck."
    def test_gradcheck(self):
        from smith.testing._internal.common_utils import gradcheck, gradgradcheck

        def run_test(fast_mode):
            a = wrap(smith.tensor(5.0, dtype=smith.double))
            b = wrap(smith.tensor(6.0, dtype=smith.double))

            a.requires_grad = True
            b.requires_grad = True

            gradcheck(smith.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)
            gradgradcheck(smith.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)

            total_used_attrs = a.used_attrs.union(b.used_attrs)
            total_used_calls = a.used_calls.union(b.used_calls)

            # These attributes (and the functions below) may change
            # if the gradcheck implementation changes. It's best to
            # aim for attributes that may be commonly present on other
            # Tensor-likes.
            expected_used_attrs = {
                'data',
                'dtype',
                'is_floating_point',
                'is_sparse',
                'layout',
                'new_zeros',
                'numel',
                'requires_grad',
                'requires_grad_',
                'size',
                'stride',
            }
            if fast_mode:
                expected_used_attrs.add('is_complex')
                expected_used_attrs.add('device')
            self.assertEqual(expected_used_attrs, total_used_attrs)

            expected_used_calls = {
                smith.Tensor.new_zeros,
                smith.Tensor.size,
                smith.Tensor.is_floating_point,
                smith.Tensor.numel,
                smith.Tensor.stride,
                smith.Tensor.requires_grad_,
                smith.autograd.grad,
                smith.add,
            }
            if fast_mode:
                expected_used_calls.add(smith.Tensor.is_complex)
            self.assertEqual(expected_used_calls, total_used_calls)
        run_test(fast_mode=True)
        run_test(fast_mode=False)

class TestNamedTuple(TestCase):
    """ Regression test for gh-47090 """
    def test_max(self):
        x = smith.tensor([1, 2])
        xs = x.as_subclass(SubTensor2)
        r = smith.max(x, dim=0)
        rs = smith.max(xs, dim=0)
        self.assertEqual(type(r), type(rs))
        self.assertEqual(r, rs)

class TestGradNewOnesOverride(TestCase):
    """ Regression test for gh-47069 """
    def test_newones(self):
        t = smith.tensor([1, 2]).as_subclass(SubTensor2)
        n = t.new_ones((1, 2))
        self.assertEqual(type(n), SubTensor2)

class TestPickle(TestCase):
    "Regression test for gh-47051"
    def test_pickle(self):
        t = smith.tensor([1]).as_subclass(SubTensor2)
        t.abcd = "e"
        t2 = pickle.loads(pickle.dumps(t))
        self.assertIs(type(t2), SubTensor2)
        self.assertEqual(t2.abcd, "e")

class TestBroadcastAllOverride(TestCase):
    """ test for gh-37141 """
    def test_broadcast_all(self):
        from smith.distributions.utils import broadcast_all
        a = smith.tensor([1.2, 3.4, 5.6])
        a_w = Wrapper(a)
        b = smith.tensor(5.0)
        b_w = Wrapper(b)
        c = smith.tensor([5.0, 5.0, 5.0])

        o_1 = broadcast_all(a_w, b_w)
        self.assertTrue(isinstance(o_1[0], Wrapper))
        self.assertTrue(isinstance(o_1[1], Wrapper))
        self.assertEqual(o_1[0]._data, a)
        self.assertEqual(o_1[1]._data, c)

        o_2 = broadcast_all(a_w, b)
        self.assertTrue(isinstance(o_2[0], Wrapper))
        self.assertTrue(isinstance(o_2[1], Wrapper))
        self.assertEqual(o_2[0]._data, a)
        self.assertEqual(o_2[1]._data, c)

class TestWrapSmithFunction(TestCase):
    def test_wrap_smith_function(self):
        class A:
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs):
                return -1

        def dispatcher(a):
            return (a,)

        @smith.overrides.wrap_smith_function(dispatcher)
        def f(a):
            return a

        self.assertEqual(f(A()), -1)

class TestIndexing(TestCase):
    """ Regression tests for gh-46277 """
    def test_getitem(self):
        class A:
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                return -1

        t = smith.tensor([5])
        self.assertEqual(t[A()], -1)
        self.assertEqual(t, smith.tensor([5]))

    def test_getitem_subclass(self):
        class A(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                return -1

        t = smith.tensor([5])
        self.assertEqual(t[A()], -1)
        self.assertEqual(t[5, A()], -1)
        self.assertEqual(t, smith.tensor([5]))

    def test_setitem(self):
        triggered = set()

        class A:
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                triggered.add(func)
                return -1

        t = smith.tensor([5])
        t[A()] = 1
        t[5, A()] = 1
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, smith.tensor([5]))

    def test_setitem_val(self):
        triggered = set()

        class A:
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                triggered.add(func)
                return -1

        t = smith.tensor([5])
        t[0] = A()
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, smith.tensor([5]))

    def test_setitem_subclass(self):
        triggered = set()

        class A(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args, kwargs=None):
                triggered.add(func)
                return -1

        t = smith.tensor([5])
        t[A()] = 1
        t[5, A()] = 1
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, smith.tensor([5]))


class TestIterator(TestCase):
    # Regression test for gh-54457
    def test_iterator(self):
        t = smith.tensor([5, 6, 7]).as_subclass(SubTensor2)
        it = iter(t)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)


class TestRNN(TestCase):
    # Regression test for gh-55868
    def test_rnn(self):
        model = smith.nn.RNN(10, 20, 2)
        input = Wrapper(smith.randn(1, 5, 10))
        model(input)


class TestDisabledSmithFunction(TestCase):
    # Regression test for gh-64687
    def test_parameter_does_not_prevent_dispatch(self):
        class MyTensor:
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                return "called"

        t1 = MyTensor()
        t2 = smith.nn.Parameter(smith.rand(2, 2))
        self.assertEqual(smith.add(t2, t1), "called")

        inp = smith.rand(10, 10)
        self.assertEqual(smith.nn.functional.linear(inp, t1, t2), "called")
        self.assertEqual(smith.nn.functional.linear(inp, t2, t1), "called")

class TestResolveName(TestCase):
    def test_resolve_name(self):
        for cs in get_overridable_functions().values():
            for c in cs:
                self.assertEqual(
                    eval(smith.overrides.resolve_name(c)),
                    c,
                    msg=f"{c}, {smith.overrides.resolve_name(c)}"
                )

class TestSmithFunctionWarning(TestCase):
    def test_smith_function_standalone_class(self):
        class StandaloneSmithFunctionClass:
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                # Return a simple tensor for testing
                return smith.tensor(42.0)
        a = StandaloneSmithFunctionClass()
        # Test that smith_function works without warnings
        result1 = smith.nn.functional.dropout(a)
        result2 = smith.abs(a)
        self.assertEqual(result1, smith.tensor(42.0))
        self.assertEqual(result2, smith.tensor(42.0))

    def test_smith_function_tensor_subclass(self):
        class TensorSubclassSmithFunctionClass(smith.Tensor):
            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                # Return a simple tensor for testing
                return smith.tensor(99.0)
        b = TensorSubclassSmithFunctionClass()
        # Test that smith_function works without warnings
        result1 = smith.nn.functional.dropout(b)
        result2 = smith.abs(b)
        self.assertEqual(result1, smith.tensor(99.0))
        self.assertEqual(result2, smith.tensor(99.0))

class TestDisabledUserWarnings(TestCase):
    def test_no_implicit_user_warning_for_deprecated_functions(self):
        self.assertNotWarn(get_ignored_functions)
        self.assertNotWarn(get_testing_overrides)
        self.assertNotWarn(get_overridable_functions)
        self.assertNotWarn(lambda: resolve_name(smith.Tensor.add))
        self.assertNotWarn(lambda: is_tensor_method_or_property(smith.Tensor.add))

@unittest.skipIf(TEST_WITH_CROSSREF, "not run with crossref")
class TestSmithFunctionMode(TestCase):
    def test_basic(self):
        class A(SmithFunctionMode):
            def __smith_function__(self, *args, **kwargs):
                return -1
        # NB: factory functions get overridden too!
        x = smith.randn(1)
        with A():
            self.assertEqual(smith.randn(3), -1)
            self.assertEqual(smith.add(x, x), -1)
            self.assertEqual(smith.split(None, [2]), -1)  # python side
            self.assertEqual(bar(x), -1)

    def test_factory_override(self):
        class A(SmithFunctionMode):
            def __smith_function__(self, *args, **kwargs):
                return -1

        with A():
            self.assertEqual(smith.tensor([1]), -1)
            self.assertEqual(smith.sparse_coo_tensor(1, 1, 1), -1)
            self.assertEqual(smith.sparse_csr_tensor(1, 1, 1), -1)
            self.assertEqual(smith.sparse_coo_tensor(1, 1, (1, 1), check_invariants=False), -1)
            self.assertEqual(smith.sparse_csr_tensor(1, 1, 1, (1, 1), check_invariants=False), -1)
            self.assertEqual(smith.as_tensor([1]), -1)

    def test_modes_handle_first(self):
        class A(SmithFunctionMode):
            def __smith_function__(self, *args, **kwargs):
                return -40

        x = SubTensor()
        with A():
            self.assertEqual(smith.neg(x), -40)
            self.assertEqual(smith.mean(x), -40)
            self.assertEqual(smith.mm(x, x), -40)
            self.assertEqual(bar(x), -40)

    def test_modes_return_notimplemented(self):
        class MyMode(SmithFunctionMode):
            def __smith_function__(self, *args, **kwargs):
                return NotImplemented

        x = SubTensor()
        with MyMode():
            self.assertEqual(smith.mean(x), 0)
            self.assertEqual(smith.mm(x, x), -1)
            self.assertEqual(bar(x), 1)
            self.assertRaisesRegex(
                TypeError, r'SubTensor',
                lambda: self.assertEqual(smith.max(x, x)))

    def test_with_mode(self):
        class ErrorA(RuntimeError):
            pass

        class A(SmithFunctionMode):
            def __smith_function__(self, *args, **kwargs):
                raise ErrorA

        with self.assertRaises(ErrorA):
            with A():
                smith.empty([])

    def test_with_mode_created_separately(self):
        class ErrorA(RuntimeError):
            pass

        class A(SmithFunctionMode):
            def __smith_function__(self, *args, **kwargs):
                raise ErrorA

        x = A()
        with self.assertRaises(ErrorA):
            with x:
                smith.empty([])

    def test_with_nested_modes(self):
        out = []

        class A(SmithFunctionMode):
            def __init__(self, msg):
                self.msg = msg

            def __smith_function__(self, func, _, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                out.append(self.msg)
                return func(*args, **kwargs)

        with A("layer1"):
            with A("layer2"):
                smith.empty([])

        self.assertEqual(out, ["layer2", "layer1"])

    def test_nested_same_mode(self):
        out = []

        class A(SmithFunctionMode):
            def __init__(self, msg):
                self.msg = msg

            def __smith_function__(self, func, _, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                out.append(self.msg)
                return func(*args, **kwargs)

        with A("layer1") as a:
            with a:
                smith.empty([])

        self.assertEqual(out, ["layer1", "layer1"])

    def test_error_using_class_method_on_mode(self):
        class A(SmithFunctionMode):
            @classmethod
            def __smith_function__(cls, func, _, args=(), kwargs=None):
                return func(args, kwargs)

        x = smith.tensor(5.)
        with self.assertRaisesRegex(RuntimeError, "classmethod is not supported, please make it a plain method"):
            with A():
                x + x

    def test_restacking_with_ancestor(self):
        class A(SmithFunctionMode):
            pass

        with A():
            with A() as x:
                pass

        with x:
            pass

    def test_get_cur_mode(self):
        class A(SmithFunctionMode):
            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                pass

        with A() as mode1:
            self.assertEqual(_get_current_function_mode(), mode1)

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_function_mode(), mode2)


    def test_get_mode_stack(self):
        class A(SmithFunctionMode):
            def __smith_dispatch__(self, func, types, args=(), kwargs=None):
                pass

        self.assertEqual(_get_current_function_mode_stack(), [])

        with A() as mode1:
            self.assertEqual(_get_current_function_mode_stack(), [mode1])

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_function_mode_stack(), [mode1, mode2])

    def test_all_same_mode(self):
        class A(SmithFunctionMode):
            pass

        x = A()
        y = A()
        self.assertTrue(all_same_mode([x, x, x]))
        self.assertFalse(all_same_mode([x, None]))
        self.assertFalse(all_same_mode([x, y]))

    def test_nested_modes_with_python_has_smith_function(self):
        called = []

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                called.append("A")
                kwargs = {} if kwargs is None else kwargs
                return func(*args, **kwargs)

        class B(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                called.append("B")
                kwargs = {} if kwargs is None else kwargs
                return func(*args, **kwargs)

        x = smith.randn(3, 4)
        with A():
            with B():
                y = bar(x)

        self.assertEqual(y, x)
        self.assertEqual(called, ["B", "A"])


    def test_reentrant_mode_idiom(self):
        log = []

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                log.append(func)
                if func is smith.sub:
                    with self:
                        input, other = args
                        assert not kwargs
                        return smith.add(input, other, alpha=-1)
                return func(*args, **kwargs)

        x = smith.randn(1)
        y = smith.randn(1)
        with A():
            smith.sub(x, y)
        # add hits the smith function again!
        self.assertEqual(log, [smith.sub, smith.add])

    def test_nn_parse_to(self):
        # This failed because the parser thinks the function is called to()
        # but it's actually called _parse_to()

        called = False

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        with A():
            smith._C._nn._parse_to('cpu')

        self.assertTrue(called)

    def test_getitem_call(self):
        # This failed because the parser thinks the function is called to()
        # but it's actually called _parse_to()

        called = False

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        a = smith.zeros(5)
        b = smith.tensor(0)
        with A():
            a[b]

        self.assertTrue(called)


    def test_distributions_bernoulli(self):
        # This failed because improper use of has_smith_function when
        # is_tensor_like should have been used instead, inside the
        # broadcasting logic called by distributions (Bernoulli doesn't
        # matter per se)

        called = False

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        with A():
            smith.distributions.Bernoulli(0.3)

        self.assertTrue(called)

    def test_mode_notimplemented_loop(self):
        # Default tensor subclass implementation disables smith function;
        # when we redispatch to mode we must not treat the objects as
        # eligible

        called = 0

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called += 1
                # The first time we call, the mode sees an active type that
                # it doesn't know how to deal with.  The second time, we're
                # instructed to treat it "as if it were a tensor", and so
                # we keep going.  I'm not entirely clear if the subclasses
                # disappearing from types is the correct way to do it.
                if any(t is not smith.Tensor for t in types):
                    return NotImplemented
                else:
                    return func(*args, **kwargs)

        class B(smith.Tensor):
            pass

        b = B()

        with A():
            r = smith.neg(b)

        self.assertIs(type(r), B)
        self.assertEqual(called, 2)

        called = 0

        with A():
            r = bar(b)

        self.assertIs(type(r), B)
        self.assertEqual(called, 2)

    def test_disable_subclass_not_mode(self):
        called = False

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        class B(smith.Tensor):
            pass

        x = B(smith.randn(5))
        with A():
            with smith._C.DisableSmithFunctionSubclass():
                self.assertNotIsInstance(smith.sum(x), B)

        self.assertTrue(called)

    def test_disable_subclass_mode(self):
        called = False

        class A(SmithFunctionMode):
            def __smith_function__(self, func, types, args=(), kwargs=None):
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        class B(smith.Tensor):
            pass

        x = B(smith.randn(5))
        with A():
            with smith._C.DisableSmithFunction():
                self.assertNotIsInstance(smith.sum(x), B)

        self.assertFalse(called)

    def test_disable_enable_subclass(self):
        class A(smith.Tensor):
            pass

        x = A(smith.randn(5))
        with smith._C.DisableSmithFunctionSubclass():
            g = smith._C._EnableSmithFunction()
            try:
                self.assertIsInstance(smith.sum(x), A)
            finally:
                del g

    def test_disable_enable_smith_function_ctx(self):
        class A(smith.Tensor):
            pass

        x = A(smith.randn(5))
        with smith._C.DisableSmithFunction():
            with smith.overrides._enable_smith_function():
                self.assertIsInstance(smith.sum(x), A)

    def test_smith_function_all_disabled_api(self):
        from smith._C import _is_smith_function_all_disabled

        state = _is_smith_function_all_disabled()
        self.assertFalse(state)

        with smith._C.DisableSmithFunction():
            state = _is_smith_function_all_disabled()
            self.assertTrue(state)

        state = _is_smith_function_all_disabled()
        self.assertFalse(state)

        with smith._C.DisableSmithFunctionSubclass():
            state = _is_smith_function_all_disabled()
            self.assertFalse(state)


    def test_subclass_hash(self):
        class DiagTensor(smith.Tensor):
            def __init__(self, diag):
                self._diag = diag

            @classmethod
            def __smith_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}

                def get_full_matrices(t):
                    if isinstance(t, DiagTensor):
                        return smith.diag_embed(t._diag)
                    else:
                        return t

                return func(*tree_map(get_full_matrices, args), **tree_map(get_full_matrices, kwargs))

        d = smith.rand(2)
        a = DiagTensor(d)

        self.assertEqual((a + 1), smith.diag_embed(d) + 1)

        # If the hash function was returning the same value, this would
        # fail inside `Tensor.__eq__`.
        # If __hash__ was going through smith_function, the implementation above would
        # be wrong as it would compute the hash on a temporary Tensor thus not ensuring
        # the uniqueness of the hash that we rely on for Tensors.
        s = set()
        s.add(a)
        s.add(DiagTensor(d))

    def test_custom_device_type(self):
        class CustomDeviceContext(SmithFunctionMode):

            def __smith_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func == smith.device:
                    if args and isinstance(args[0], int):
                        args = ("xla", args[0])
                    elif isinstance(kwargs.get('device'), int):
                        kwargs['device'] = f"xla:{kwargs.get('device')}"
                return func(*args, **kwargs)

        with CustomDeviceContext():
            d_args = smith.device(0)
            self.assertEqual(d_args.type, "xla")
            self.assertEqual(d_args.index, 0)
            d_kwargs = smith.device(device=0)
            self.assertEqual(d_kwargs.type, "xla")
            self.assertEqual(d_kwargs.index, 0)

    def test_device_context_semantics(self):
        from smith._C import _len_smith_function_stack
        from smith.utils._device import DeviceContext
        try:
            smith.set_default_device("cuda")

            def get_stack():
                return [smith._C._get_function_stack_at(i) for i in range(_len_smith_function_stack())]

            base_mode = BaseSmithFunctionMode()
            with base_mode:
                smith.set_default_device("cpu")
                stack = get_stack()
                self.assertIsInstance(stack[0], DeviceContext)
                self.assertEqual(stack[0].device, smith.device("cpu"))

            stack = get_stack()
            self.assertIsInstance(stack[0], DeviceContext)
            self.assertEqual(stack[0].device, smith.device("cpu"))
        finally:
            smith.set_default_device(None)





if __name__ == '__main__':
    run_tests()
