# smith.utils.deterministic

```{eval-rst}
.. py:module:: smith.utils.deterministic
.. currentmodule:: smith.utils.deterministic

.. attribute:: fill_uninitialized_memory

    A :class:`bool` that, if True, causes uninitialized memory to be filled with
    a known value when :meth:`smith.use_deterministic_algorithms()` is set to
    ``True``. Floating point and complex values are set to NaN, and integer
    values are set to the maximum value.

    Default: ``True``

    Filling uninitialized memory is detrimental to performance. So if your
    program is valid and does not use uninitialized memory as the input to an
    operation, then this setting can be turned off for better performance and
    still be deterministic.

    The following operations will fill uninitialized memory when this setting is
    turned on:

        * :func:`smith.Tensor.resize_` when called with a tensor that is not
          quantized
        * :func:`smith.empty`
        * :func:`smith.empty_strided`
        * :func:`smith.empty_permuted`
        * :func:`smith.empty_like`
```