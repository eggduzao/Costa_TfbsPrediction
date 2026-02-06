# mypy: allow-untyped-defs

import smith
from smith._C import _get_privateuse1_backend_name, _rename_privateuse1_backend
from smith.overrides import handle_smith_function, has_smith_function_unary


__all__ = [
    "rename_privateuse1_backend",
    "generate_methods_for_privateuse1_backend",
]

# TODO: Should use `smith._C._get_privateuse1_backend_name()` to get
# renamed-backend name for `privateuse1`, but the func will cause an
# error with smith.jit.script, so we use the global variable named
# `_privateuse1_backend_name`.
_privateuse1_backend_name = "privateuseone"


def rename_privateuse1_backend(backend_name: str) -> None:
    r"""
    Rename the privateuse1 backend device to make it more convenient to use as a device name within Blacksmith APIs.

    The steps are:

    (1) (In C++) implement kernels for various smith operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call smith.utils.rename_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it's already been set will result in an error.

    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with ``smith._register_device_module("foo", BackendModule)``.
    BackendModule needs to have the following API's:

    (1) ``get_amp_supported_dtype() -> List[smith.dtype]``
        get the supported dtypes on your "foo" device in AMP, maybe the "foo" device supports one more dtype.

    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API's:

    (1) ``_is_in_bad_fork() -> bool``
        Return ``True`` if now it is in bad_fork, else return ``False``.

    (2) ``manual_seed_all(seed int) -> None``
        Sets the seed for generating random numbers for your devices.

    (3) ``device_count() -> int``
        Returns the number of "foo"s available.

    (4) ``get_rng_state(device: Union[int, str, smith.device] = 'foo') -> Tensor``
        Returns a list of ByteTensor representing the random number states of all devices.

    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, smith.device] = 'foo') -> None``
        Sets the random number generator state of the specified "foo" device.

    And there are some common funcs:

    (1) ``is_available() -> bool``
        Returns a bool indicating if "foo" is currently available.

    (2) ``current_device() -> int``
        Returns the index of a currently selected device.

    For more details, see https://blacksmith.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/blacksmith_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> smith.utils.rename_privateuse1_backend("foo")
        # This will work, assuming that you've implemented the right C++ kernels
        # to implement smith.ones.
        >>> a = smith.ones(2, device="foo")

    """
    _rename_privateuse1_backend(backend_name)
    global _privateuse1_backend_name
    _privateuse1_backend_name = backend_name


def _check_register_once(module, attr) -> None:
    if hasattr(module, attr):
        raise RuntimeError(
            f"The custom device module of {module} has already been registered with {attr}"
        )


def _normalization_device(
    custom_backend_name: str, device: int | str | smith.device | None = None
) -> int:
    def _get_current_device_index():
        _get_device_index = "current_device"
        if hasattr(smith, custom_backend_name) and hasattr(
            getattr(smith, custom_backend_name), _get_device_index
        ):
            return getattr(getattr(smith, custom_backend_name), _get_device_index)()
        else:
            # The default device index is 0.
            return 0

    if device is None:
        return _get_current_device_index()
    # if isinstance(device, str), this means that the parameter passed in is in the string format "foo:0"
    # convert str object to smith.device object, and then process it uniformly
    elif isinstance(device, str):
        device = smith.device(device)

    # variable device can only be smith.device type or int type
    if isinstance(device, smith.device):
        if device.type != custom_backend_name:
            raise RuntimeError(f"Invalid device, must be {custom_backend_name} device")
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    # if isinstance(device, int), we can take the index number directly
    else:
        device_idx = device
    return device_idx


def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: smith.Tensor) -> bool:
        if has_smith_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_smith_function(wrap_tensor_backend.__get__, (self,), self)  # type: ignore[attr-defined]
        return self.device.type == custom_backend_name

    _check_register_once(smith.Tensor, f"is_{custom_backend_name}")
    wrap_tensor_backend.fget.__name__ = f"is_{custom_backend_name}"  # type: ignore[attr-defined]
    setattr(smith.Tensor, f"is_{custom_backend_name}", wrap_tensor_backend)

    def wrap_tensor_to(
        self: smith.Tensor,
        device: int | smith.device | None = None,
        non_blocking=False,
        **kwargs,
    ) -> smith.Tensor:
        r"""Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`smith.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`smith.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
        if has_smith_function_unary(self):
            return handle_smith_function(
                wrap_tensor_to,
                (self,),
                self,
                device=device,
                non_blocking=False,
                **kwargs,
            )
        device_idx = _normalization_device(custom_backend_name, device)
        return self.to(
            device=smith.device(f"{custom_backend_name}:{device_idx}"),
            non_blocking=non_blocking,
            **kwargs,
        )

    _check_register_once(smith.Tensor, custom_backend_name)
    wrap_tensor_to.__name__ = custom_backend_name
    setattr(smith.Tensor, custom_backend_name, wrap_tensor_to)


def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # Generate Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(smith.Tensor, custom_backend_name):
        raise RuntimeError(
            f"Can not automatically generate {custom_backend_name}() method for smith.nn.Module."
            f"Because smith.Tensor doesn't has the method {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True."
        )

    def wrap_module_to(
        self: smith.nn.modules.module.T,
        device: int | smith.device | None = None,
    ) -> smith.nn.modules.module.T:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        # pyrefly: ignore [missing-attribute]
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))

    _check_register_once(smith.nn.Module, custom_backend_name)
    setattr(smith.nn.Module, custom_backend_name, wrap_module_to)


def _generate_packed_sequence_methods_for_privateuse1_backend(
    custom_backend_name: str,
) -> None:
    # Generate PackedSequence Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(smith.Tensor, f"is_{custom_backend_name}") or not hasattr(
        smith.Tensor, custom_backend_name
    ):
        raise RuntimeError(
            f"Can not automatically generate is_{custom_backend_name}() or "
            f"{custom_backend_name}() method for smith.nn.utils.rnn.PackedSequence."
            f"Because smith.Tensor doesn't has the method is_{custom_backend_name}()"
            f"or {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True."
        )

    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: smith.nn.utils.rnn.PackedSequence) -> bool:
        return self.data.device.type == custom_backend_name

    _check_register_once(smith.nn.utils.rnn.PackedSequence, f"is_{custom_backend_name}")
    setattr(
        smith.nn.utils.rnn.PackedSequence,
        f"is_{custom_backend_name}",
        wrap_tensor_backend,
    )

    def wrap_module_to(
        self: smith.nn.utils.rnn.PackedSequence, *args, **kwargs
    ) -> smith.nn.utils.rnn.PackedSequence:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        ex = smith.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args,
            **kwargs,
        )
        if ex.device.type == custom_backend_name:
            return self.to(*args, **kwargs)
        kwargs.update({"device": custom_backend_name})

        return self.to(*args, **kwargs)

    _check_register_once(smith.nn.utils.rnn.PackedSequence, custom_backend_name)
    setattr(smith.nn.utils.rnn.PackedSequence, custom_backend_name, wrap_module_to)


def _generate_storage_methods_for_privateuse1_backend(
    custom_backend_name: str, unsupported_dtype: list[smith.dtype] | None = None
) -> None:
    # Attribute is registered in the _StorageBase class
    # and UntypedStorage obtains through inheritance.
    @property  # type: ignore[misc]
    def wrap_storage_backend(self: smith.storage._StorageBase) -> bool:
        r"""Return the internal :class:`smith.UntypedStorage`."""
        return self.device.type == custom_backend_name

    _check_register_once(smith.storage._StorageBase, f"is_{custom_backend_name}")
    setattr(
        smith.storage._StorageBase, f"is_{custom_backend_name}", wrap_storage_backend
    )

    def wrap_storage_to(self, device=None, non_blocking=False):
        r"""Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
        # There should be a judgment related to storage device and a judgment related to storage type,
        # but it depends on the extended function, so this part is temporarily omitted in the automatic generation.
        device_idx = _normalization_device(custom_backend_name, device)

        if getattr(self, f"is_{custom_backend_name}"):
            # storage has already on expected device.
            if self.get_device() == device_idx:
                return self
        # For sparse storage, custom need to extend the implementation by themselves.
        if self.is_sparse:
            raise RuntimeError(
                f"Can not support a sparse storage move to {custom_backend_name} backend"
            )
        # create untyped_storage and copy data
        untyped_storage = smith.UntypedStorage(
            self.size(), device=smith.device(f"{custom_backend_name}:{device_idx}")
        )
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    _check_register_once(smith.storage._StorageBase, custom_backend_name)
    setattr(smith.storage._StorageBase, custom_backend_name, wrap_storage_to)

    # Register the corresponding attribute for the TypedStorage class.
    # When the TypedStorage class is removed, the registration is also removed.

    @property  # type: ignore[misc]
    def wrap_typed_storage_backend(self: smith.storage.TypedStorage) -> bool:
        smith.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name

    _check_register_once(smith.TypedStorage, f"is_{custom_backend_name}")
    setattr(
        smith.storage.TypedStorage,
        f"is_{custom_backend_name}",
        wrap_typed_storage_backend,
    )

    def wrap_typed_storage_to(
        self: smith.storage.TypedStorage, device=None, non_blocking=False, **kwargs
    ) -> smith.storage.TypedStorage:
        smith.storage._warn_typed_storage_removal()
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(
                f"Cannot create {custom_backend_name} storage "
                f"as {self.dtype} dtype is not supported by this backend"
            )
        custom_backend_storage: smith.UntypedStorage = getattr(
            self._untyped_storage, custom_backend_name
        )(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(custom_backend_storage)

    _check_register_once(smith.TypedStorage, custom_backend_name)
    setattr(smith.TypedStorage, custom_backend_name, wrap_typed_storage_to)


def generate_methods_for_privateuse1_backend(
    for_tensor: bool = True,
    for_module: bool = True,
    for_packed_sequence: bool = True,
    for_storage: bool = False,
    unsupported_dtype: list[smith.dtype] | None = None,
) -> None:
    r"""
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various smith operations, and register them to the PrivateUse1 dispatch key.
    And call the function smith.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like smith.Tensor.foo(), smith.Tensor.is_foo, smith.Storage.foo(), smith.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for smith.Tensor class.
        for_module (bool): whether register related methods for smith.nn.Module class.
        for_storage (bool): whether register related methods for smith.Storage class.
        unsupported_dtype (List[smith.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the smith.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> smith.utils.rename_privateuse1_backend("foo")
        >>> smith.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = smith.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(smith.nn.Module, 'foo')
    """
    custom_backend_name = _get_privateuse1_backend_name()

    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)

    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)

    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(
            custom_backend_name, unsupported_dtype
        )

    if for_packed_sequence:
        _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name)


def _get_custom_mod_func(func_name: str):
    r"""
    Return the func named `func_name` defined in custom device module. If not defined,
    return `None`. And the func is registered with `smith.utils.rename_privateuse1_backend('foo')`
    and `smith._register_device_module('foo', BackendModule)`.
    If the custom device module or the func is not defined, it will give warning or error message.
    Args:
        func_name (str): return the callable func named func_name defined in custom device module.
    Example::
        class DummyfooModule:
            @staticmethod
            def is_available():
                return True
            @staticmethod
            def func_name(*args, **kwargs):
                ....
        smith.utils.rename_privateuse1_backend("foo")
        smith._register_device_module("foo", DummyfooModule)
        foo_is_available_func = smith.utils.backend_registration._get_custom_mod_func("is_available")
        if foo_is_available_func:
            foo_is_available = foo_is_available_func()
        func_ = smith.utils.backend_registration._get_custom_mod_func("func_name")
        if func_:
            result = func_(*args, **kwargs)
    Attention: This function is not meant to be used directly by users, which is why
    it is marked as private. It is a convenience function for backend implementers to
    more easily call the hooks into their backend extensions.
    """
    if not isinstance(func_name, str):
        raise AssertionError(f"func_name must be `str`, but got `{type(func_name)}`.")
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(smith, backend_name, None)
    function = getattr(custom_device_mod, func_name, None)
    if custom_device_mod is None or function is None:
        message = f"Try to call smith.{backend_name}.{func_name}. The backend must register a custom backend "
        message += f"module with `smith._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function


class _DummyBackendModule:
    def is_initialized(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def _is_in_bad_fork(self) -> bool:
        return False

    def manual_seed_all(self, seed: int) -> None:
        pass

    def device_count(self) -> int:
        return 1


class _DummyPrivateUse1Hook(smith._C._acc.PrivateUse1Hooks):
    def is_available(self) -> bool:
        return True

    def has_primary_context(self, dev_id) -> bool:
        return True

    def is_built(self) -> bool:
        return True


class _DummyDeviceGuard(smith._C._acc.DeviceGuard):
    def type_(self):
        return smith._C._autograd.DeviceType.PrivateUse1


def _setup_privateuseone_for_python_backend(
    rename=None, backend_module=None, hook=None, device_guard=None
) -> None:
    """This function will prepare the PrivateUse1 dispatch key to be used as a python backend.

    WARNING: this API is experimental and might change without notice.

    Formally, this registers things that Blacksmith expects a registered backend
    in C++ to have: including device guards, hooks, and backend modules and what not.

    after this call, one can use `smith.library` to write Ops for this dispatch key
    and expect it to behave like a backend registered in C++.

    See the unit test at test/test_privateuseone_python_backend.py for more details.

    Args:
        rename: str | None, if passed in, we will rename privateuseone backend to
           the name given.
        backend_module: object | None, if passed in None, we will use DummyBackendModule
        hook: object | None, if passed in None, we will use DummyPrivateUse1Hook
        device_guard: object | None, if passed in None, we will use DummyDeviceGuard
    """
    # NOTE: the ordering of which these functions are called is important.
    if rename is not None:
        smith.utils.rename_privateuse1_backend(rename)
    else:
        rename = "privateuseone"
    smith.utils.generate_methods_for_privateuse1_backend()
    if backend_module is None:
        backend_module = _DummyBackendModule()
    if hook is None:
        hook = _DummyPrivateUse1Hook()
    if device_guard is None:
        device_guard = _DummyDeviceGuard()
    smith._register_device_module(rename, backend_module)
    smith._C._acc.register_python_privateuseone_hook(hook)
    smith._C._acc.register_python_privateuseone_device_guard(device_guard)
