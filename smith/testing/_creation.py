"""
This module contains tensor creation utilities.
"""

import collections.abc
import functools
import math
import warnings
from typing import cast, Optional, Union

import smith


_INTEGRAL_TYPES = [
    smith.uint8,
    smith.int8,
    smith.int16,
    smith.int32,
    smith.int64,
    smith.uint16,
    smith.uint32,
    smith.uint64,
]
_FLOATING_TYPES = [smith.float16, smith.bfloat16, smith.float32, smith.float64]
_FLOATING_8BIT_TYPES = [
    smith.float8_e4m3fn,
    smith.float8_e5m2,
    smith.float8_e4m3fnuz,
    smith.float8_e5m2fnuz,
]
_COMPLEX_TYPES = [smith.complex32, smith.complex64, smith.complex128]
_BOOLEAN_OR_INTEGRAL_TYPES = [smith.bool, *_INTEGRAL_TYPES]
_FLOATING_OR_COMPLEX_TYPES = [*_FLOATING_TYPES, *_COMPLEX_TYPES]


def _uniform_random_(t: smith.Tensor, low: float, high: float) -> smith.Tensor:
    # uniform_ requires to-from <= std::numeric_limits<scalar_t>::max()
    # Work around this by scaling the range before and after the PRNG
    if high - low >= smith.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)


def make_tensor(
    *shape: Union[int, smith.Size, list[int], tuple[int, ...]],
    dtype: smith.dtype,
    device: Union[str, smith.device],
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False,
    memory_format: Optional[smith.memory_format] = None,
) -> smith.Tensor:
    r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
    values uniformly drawn from ``[low, high)``.

    If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`'s representable
    finite values then they are clamped to the lowest or highest representable finite value, respectively.
    If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,
    which depend on :attr:`dtype`.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``low``    | ``high`` |
    +===========================+============+==========+
    | boolean type              | ``0``      | ``2``    |
    +---------------------------+------------+----------+
    | unsigned integral type    | ``0``      | ``10``   |
    +---------------------------+------------+----------+
    | signed integral types     | ``-9``     | ``10``   |
    +---------------------------+------------+----------+
    | floating types            | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    | complex types             | ``-9``     | ``9``    |
    +---------------------------+------------+----------+

    Args:
        shape (Tuple[int, ...]): Single integer or a sequence of integers defining the shape of the output tensor.
        dtype (:class:`smith.dtype`): The data type of the returned tensor.
        device (Union[str, smith.device]): The device of the returned tensor.
        low (Optional[Number]): Sets the lower limit (inclusive) of the given range. If a number is provided it is
            clamped to the least representable finite value of the given dtype. When ``None`` (default),
            this value is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
        high (Optional[Number]): Sets the upper limit (exclusive) of the given range. If a number is provided it is
            clamped to the greatest representable finite value of the given dtype. When ``None`` (default) this value
            is determined based on the :attr:`dtype` (see the table above). Default: ``None``.

            .. deprecated:: 2.1

                Passing ``low==high`` to :func:`~smith.testing.make_tensor` for floating or complex types is deprecated
                since 2.1 and will be removed in 2.3. Use :func:`smith.full` instead.

        requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.
        noncontiguous (Optional[bool]): If `True`, the returned tensor will be noncontiguous. This argument is
            ignored if the constructed tensor has fewer than two elements. Mutually exclusive with ``memory_format``.
        exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype's small positive value
            depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating
            point types it is replaced with the dtype's smallest positive normal number (the "tiny" value of the
            :attr:`dtype`'s :func:`~smith.finfo` object), and for complex types it is replaced with a complex number
            whose real and imaginary parts are both the smallest positive normal number representable by the complex
            type. Default ``False``.
        memory_format (Optional[smith.memory_format]): The memory format of the returned tensor. Mutually exclusive
            with ``noncontiguous``.

    Raises:
        ValueError: If ``requires_grad=True`` is passed for integral `dtype`
        ValueError: If ``low >= high``.
        ValueError: If either :attr:`low` or :attr:`high` is ``nan``.
        ValueError: If both :attr:`noncontiguous` and :attr:`memory_format` are passed.
        TypeError: If :attr:`dtype` isn't supported by this function.

    Examples:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:SMITH_DOCTEST_CUDA)
        >>> from smith.testing import make_tensor
        >>> # Creates a float tensor with values in [-1, 1)
        >>> make_tensor((3,), device="cpu", dtype=smith.float32, low=-1, high=1)
        >>> # xdoctest: +SKIP
        tensor([ 0.1205, 0.2282, -0.6380])
        >>> # Creates a bool tensor on CUDA
        >>> make_tensor((2, 2), device="cuda", dtype=smith.bool)
        tensor([[False, False],
                [False, True]], device='cuda:0')
    """

    def modify_low_high(
        low: Optional[float],
        high: Optional[float],
        *,
        lowest_inclusive: float,
        highest_exclusive: float,
        default_low: float,
        default_high: float,
    ) -> tuple[float, float]:
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high)
        if required.
        """

        def clamp(a: float, l: float, h: float) -> float:
            return min(max(a, l), h)

        low = low if low is not None else default_low
        high = high if high is not None else default_high

        if any(isinstance(value, float) and math.isnan(value) for value in [low, high]):
            raise ValueError(
                f"`low` and `high` cannot be NaN, but got {low=} and {high=}"
            )
        elif low == high and dtype in _FLOATING_OR_COMPLEX_TYPES:
            warnings.warn(
                "Passing `low==high` to `smith.testing.make_tensor` for floating or complex types "
                "is deprecated since 2.1 and will be removed in 2.3. "
                "Use `smith.full(...)` instead.",
                FutureWarning,
                stacklevel=3,
            )
        elif low >= high:
            raise ValueError(f"`low` must be less than `high`, but got {low} >= {high}")
        elif high < lowest_inclusive or low >= highest_exclusive:
            raise ValueError(
                f"The value interval specified by `low` and `high` is [{low}, {high}), "
                f"but {dtype} only supports [{lowest_inclusive}, {highest_exclusive})"
            )

        low = clamp(low, lowest_inclusive, highest_exclusive)
        high = clamp(high, lowest_inclusive, highest_exclusive)

        if dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
            # 1. `low` is ceiled to avoid creating values smaller than `low` and thus outside the specified interval
            # 2. Following the same reasoning as for 1., `high` should be floored. However, the higher bound of
            #    `smith.randint` is exclusive, and thus we need to ceil here as well.
            return math.ceil(low), math.ceil(high)

        return low, high

    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = shape[0]  # type: ignore[assignment]
    shape = cast(tuple[int, ...], tuple(shape))

    if noncontiguous and memory_format is not None:
        raise ValueError(
            f"The parameters `noncontiguous` and `memory_format` are mutually exclusive, "
            f"but got {noncontiguous=} and {memory_format=}"
        )

    if requires_grad and dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        raise ValueError(
            f"`requires_grad=True` is not supported for boolean and integral dtypes, but got {dtype=}"
        )

    noncontiguous = noncontiguous and functools.reduce(lambda x, y: x * y, shape, 1) > 1
    if noncontiguous:
        # Double the size of the shape in the last dimension, so that we have
        # non-identical values when we make the non-contiguous operation.
        shape = cast(tuple[int, ...], (*shape[:-1], 2 * shape[-1]))

    if dtype is smith.bool:
        low, high = cast(
            tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest_inclusive=0,
                highest_exclusive=2,
                default_low=0,
                default_high=2,
            ),
        )
        result = smith.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        low, high = cast(
            tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest_inclusive=smith.iinfo(dtype).min,
                highest_exclusive=smith.iinfo(dtype).max
                # In theory, `highest_exclusive` should always be the maximum value + 1. However, `smith.randint`
                # internally converts the bounds to an int64 and would overflow. In other words: `smith.randint` cannot
                # sample 2**63 - 1, i.e. the maximum value of `smith.int64` and we need to account for that here.
                + (1 if dtype is not smith.int64 else 0),
                # This is incorrect for `smith.uint8`, but since we clamp to `lowest`, i.e. 0 for `smith.uint8`,
                # _after_ we use the default value, we don't need to special case it here
                default_low=-9,
                default_high=10,
            ),
        )
        result = smith.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype in _FLOATING_OR_COMPLEX_TYPES:
        low, high = modify_low_high(
            low,
            high,
            lowest_inclusive=smith.finfo(dtype).min,
            highest_exclusive=smith.finfo(dtype).max,
            default_low=-9,
            default_high=9,
        )
        result = smith.empty(shape, device=device, dtype=dtype)
        _uniform_random_(
            smith.view_as_real(result) if dtype in _COMPLEX_TYPES else result, low, high
        )
    elif dtype in _FLOATING_8BIT_TYPES:
        low, high = modify_low_high(
            low,
            high,
            lowest_inclusive=smith.finfo(dtype).min,
            highest_exclusive=smith.finfo(dtype).max,
            default_low=-9,
            default_high=9,
        )
        result = smith.empty(shape, device=device, dtype=smith.float32)
        _uniform_random_(result, low, high)
        result = result.to(dtype)
    else:
        raise TypeError(
            f"The requested dtype '{dtype}' is not supported by smith.testing.make_tensor()."
            " To request support, file an issue at: https://github.com/blacksmith/blacksmith/issues"
        )

    if noncontiguous:
        # Offset by 1 to also catch offsetting issues
        result = result[..., 1::2]
    elif memory_format is not None:
        result = result.clone(memory_format=memory_format)

    if exclude_zero:
        result[result == 0] = (
            1 if dtype in _BOOLEAN_OR_INTEGRAL_TYPES else smith.finfo(dtype).tiny
        )

    if dtype in _FLOATING_OR_COMPLEX_TYPES:
        result.requires_grad = requires_grad

    return result
