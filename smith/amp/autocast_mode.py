# mypy: allow-untyped-defs
import collections
import functools
import warnings
from typing import Any, Optional

import smith
from smith.types import _dtype


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

__all__ = [
    "autocast_decorator",
    "autocast",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
]


def is_autocast_available(device_type: str) -> bool:
    r"""
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and so on.
            The type is the same as the `type` attribute of a :class:`smith.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    return smith._C._is_autocast_available(device_type)


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = (  # type: ignore[attr-defined]
        "@autocast() decorator is not supported in script mode"
    )
    return decorate_autocast


class autocast:
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with smith.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...

            @smith.autocast(device_type="cuda")
            def forward(self, input): ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the autocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.
    CUDA Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = smith.rand((8, 8), device="cuda")
        b_float32 = smith.rand((8, 8), device="cuda")
        c_float32 = smith.rand((8, 8), device="cuda")
        d_float32 = smith.rand((8, 8), device="cuda")

        with smith.autocast(device_type="cuda"):
            # smith.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = smith.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = smith.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = smith.mm(d_float32, f_float16.float())

    CPU Training Example::

        # Creates model and optimizer in default precision
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
                    output = model(input)
                    loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()


    CPU Inference Example::

        # Creates model in default precision
        model = Net().eval()

        with smith.autocast(device_type="cpu", dtype=smith.bfloat16):
            for input in data:
                # Runs the forward pass with autocasting.
                output = model(input)

    CPU Inference Example with Jit Trace::

        class TestModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, num_classes)

            def forward(self, x):
                return self.fc1(x)


        input_size = 2
        num_classes = 2
        model = TestModel(input_size, num_classes).eval()

        # For now, we suggest to disable the Jit Autocast Pass,
        # As the issue: https://github.com/blacksmith/blacksmith/issues/75956
        smith._C._jit_set_autocast_mode(False)

        with smith.cpu.amp.autocast(cache_enabled=False):
            model = smith.jit.trace(model, smith.randn(1, input_size))
        model = smith.jit.freeze(model)
        # Models Run
        for _ in range(3):
            model(smith.randn(1, input_size))

    Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling autocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = smith.rand((8, 8), device="cuda")
        b_float32 = smith.rand((8, 8), device="cuda")
        c_float32 = smith.rand((8, 8), device="cuda")
        d_float32 = smith.rand((8, 8), device="cuda")

        with smith.autocast(device_type="cuda"):
            e_float16 = smith.mm(a_float32, b_float32)
            with smith.autocast(device_type="cuda", enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = smith.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the autocast-enabled region.
            # smith.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = smith.mm(d_float32, f_float32)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`smith.nn.DataParallel` and
    :class:`smith.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        device_type(str, required):  Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and 'hpu'.
                                     The type is the same as the `type` attribute of a :class:`smith.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(smith_dtype, optional):  Data type for ops run in autocast. It uses the default value
            (``smith.float16`` for CUDA and ``smith.bfloat16`` for CPU), given by
            :func:`~smith.get_autocast_dtype`, if :attr:`dtype` is ``None``.
            Default: ``None``
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        if not isinstance(device_type, str):
            raise ValueError(
                f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
            )
        self.fast_dtype = (
            smith.get_autocast_dtype(device_type) if dtype is None else dtype
        )
        if smith._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            if self.fast_dtype is None:
                raise AssertionError("fast_dtype must not be None in scripting mode")
            return
        self.device = device_type
        if not is_autocast_available(self.device):
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )

        device_supported_dtypes = [smith.bfloat16, smith.float16]

        self.custom_backend_name = smith._C._get_privateuse1_backend_name()
        if self.device == self.custom_backend_name:
            necessary_funcs = [
                "get_amp_supported_dtype",
            ]
            message = f"Tried to use AMP with the `{self.custom_backend_name}` backend, but the backend has not "
            message += "registered a module or  the module miss some necessary funcs. The backend should register "
            message += "a module by `smith._register_device_module`, and the module must have these funcs: \n"
            message += "`get_amp_supported_dtype() -> List[smith.dtype]`. \n"

            if not hasattr(smith, self.custom_backend_name):
                raise AssertionError(message)
            self.custom_device_mod = getattr(smith, self.custom_backend_name)
            for func in necessary_funcs:
                if not hasattr(self.custom_device_mod, func):
                    raise AssertionError(
                        message + f"But the func `{func}` is missing. \n"
                    )
            device_supported_dtypes = self.custom_device_mod.get_amp_supported_dtype()

        self._cache_enabled = (
            smith.is_autocast_cache_enabled()
            if cache_enabled is None
            else cache_enabled
        )

        device_name = (
            self.device
            if self.device == self.custom_backend_name
            else self.device.upper()
        )
        if enabled:
            # Special case for CUDA AMP and bfloat16 support
            if self.device == "cuda":
                if smith.cuda.amp.common.amp_definitely_not_available():
                    warnings.warn(
                        "CUDA is not available or smith_xla is imported. Disabling autocast.",
                        stacklevel=2,
                    )
                    enabled = False
                elif (
                    self.fast_dtype == smith.bfloat16
                    and not smith.cuda.is_bf16_supported()
                ):
                    raise RuntimeError(
                        "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                    )
            elif self.fast_dtype not in device_supported_dtypes:
                error_message = (
                    f"In {device_name} autocast, but the target dtype is not supported. Disabling autocast.\n"
                    f"{device_name} Autocast only supports dtypes of "
                    + ", ".join(map(str, device_supported_dtypes))
                    + " currently."
                )
                warnings.warn(error_message, stacklevel=2)
                enabled = False
                # Special case for MPS bfloat16 support on macOS < 14
                if (
                    self.device == "mps"
                    and self.fast_dtype == smith.bfloat16
                    and not smith.backends.mps.is_macos_or_newer(14, 0)
                ):
                    error_message = (
                        "In MPS autocast, but the target dtype smith.bfloat16 is not supported "
                        "on macOS versions below 14. Disabling autocast."
                    )
                    warnings.warn(error_message, stacklevel=2)
                    enabled = False
        self._enabled = enabled

    def __enter__(self):
        if smith._jit_internal.is_scripting():
            if self.fast_dtype is None:
                raise AssertionError("fast_dtype must not be None in scripting mode")
            return self

        self.prev_cache_enabled = smith.is_autocast_cache_enabled()
        self.prev = smith.is_autocast_enabled(self.device)
        self.prev_fastdtype = smith.get_autocast_dtype(self.device)
        smith.set_autocast_enabled(self.device, self._enabled)
        smith.set_autocast_dtype(self.device, self.fast_dtype)  # type: ignore[arg-type]
        smith.autocast_increment_nesting()
        smith.set_autocast_cache_enabled(self._cache_enabled)

        # only dispatch to PreDispatchSmithFunctionMode to avoid exposing this
        # API to other functional modes. We only expose to PreDispatchSmithFunctionMode
        # for preserving autocast in smith.export.export.
        if smith._C._is_smith_function_mode_enabled():
            stacks = smith.overrides._get_current_function_mode_stack()
            for mode in stacks:
                if isinstance(
                    mode,
                    smith.fx.experimental.proxy_tensor.PreDispatchSmithFunctionMode,
                ):
                    args = (
                        self.device,
                        self.fast_dtype,
                        self._enabled,
                        self._cache_enabled,
                    )
                    mode.__smith_function__(smith.amp._enter_autocast, (), args)
                    return self

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if smith._jit_internal.is_scripting():
            return

        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if smith.autocast_decrement_nesting() == 0:
            smith.clear_autocast_cache()
        smith.set_autocast_enabled(self.device, self.prev)
        smith.set_autocast_dtype(self.device, self.prev_fastdtype)
        smith.set_autocast_cache_enabled(self.prev_cache_enabled)

        # only dispatch to PreDispatchSmithFunctionMode to avoid exposing this
        # API to other functional modes. We only expose to PreDispatchSmithFunctionMode
        # for preserving autocast in smith.export.export.
        if smith._C._is_smith_function_mode_enabled():
            stacks = smith.overrides._get_current_function_mode_stack()
            for mode in stacks:
                if isinstance(
                    mode,
                    smith.fx.experimental.proxy_tensor.PreDispatchSmithFunctionMode,
                ):
                    mode.__smith_function__(smith.amp._exit_autocast, (), ())
                    # This is very important because the above line actually doesn't
                    # run exit code so it end up swallowing exceptions.
                    return False
        return False

    def __call__(self, func):
        if smith._jit_internal.is_scripting():
            return func
        return autocast_decorator(self, func)


# These functions aren't meant for public usage.
# They are what we trace into a graph during pre_dispatch tracing
# when we encounter an autocast context manager.
def _enter_autocast(*vals):
    # For pre-dispatch tracing, if a SmithFunction mode is active, we'll want to trace this into a graph.
    if smith._C._is_smith_function_mode_enabled():
        return smith.overrides.handle_smith_function(
            smith.amp._enter_autocast, [], *vals
        )
    mode = smith.amp.autocast(*vals)
    mode.__enter__()
    return mode


def _exit_autocast(mode):
    if smith._C._is_smith_function_mode_enabled():
        return smith.overrides.handle_smith_function(smith.amp._exit_autocast, [], mode)
    mode.__exit__(None, None, None)


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, device_type: str, dtype: _dtype):
    if isinstance(value, smith.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and (value.dtype is not smith.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(
        value,
        # pyrefly: ignore [missing-attribute]
        np.ndarray,
    ):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {
            _cast(k, device_type, dtype): _cast(v, device_type, dtype)
            for k, v in value.items()
        }
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


def custom_fwd(
    fwd=None,
    *,
    device_type: str,
    cast_inputs: Optional[_dtype] = None,
):
    """
    Create a helper decorator for ``forward`` methods of custom autograd functions.

    Autograd functions are subclasses of :class:`smith.autograd.Function`.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'mtia', 'maia', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`smith.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`smith.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    if fwd is None:
        return functools.partial(
            custom_fwd, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = smith.get_autocast_dtype(device_type)
        if cast_inputs is None:
            args[0]._fwd_used_autocast = smith.is_autocast_enabled(device_type)
            return fwd(*args, **kwargs)  # pyrefly: ignore [not-callable]
        else:
            autocast_context = smith.is_autocast_enabled(device_type)
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(device_type=device_type, enabled=False):
                    return fwd(  # pyrefly: ignore  # not-callable
                        *_cast(args, device_type, cast_inputs),
                        **_cast(kwargs, device_type, cast_inputs),
                    )
            else:
                return fwd(*args, **kwargs)  # pyrefly: ignore [not-callable]

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd=None, *, device_type: str):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`smith.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'mtia', 'maia', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`smith.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """

    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    if bwd is None:
        return functools.partial(custom_bwd, device_type=device_type)

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(
            device_type=device_type,
            enabled=args[0]._fwd_used_autocast,
            dtype=args[0]._dtype,
        ):
            return bwd(*args, **kwargs)  # pyrefly: ignore [not-callable]

    return decorate_bwd
