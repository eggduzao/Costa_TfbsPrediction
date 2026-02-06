(smith-library-docs)=

# smith.library

```{eval-rst}
.. py:module:: smith.library
.. currentmodule:: smith.library
```

smith.library is a collection of APIs for extending Blacksmith's core library
of operators. It contains utilities for testing custom operators, creating new
custom operators, and extending operators defined with Blacksmith's C++ operator
registration APIs (e.g. aten operators).

For a detailed guide on effectively using these APIs, please see
[Blacksmith Custom Operators Landing Page](https://blacksmith.org/tutorials/advanced/custom_ops_landing_page.html)
for more details on how to effectively use these APIs.

## Testing custom ops

Use {func}`smith.library.opcheck` to test custom ops for incorrect usage of the
Python smith.library and/or C++ SMITH_LIBRARY APIs. Also, if your operator supports
training, use {func}`smith.autograd.gradcheck` to test that the gradients are
mathematically correct.

```{eval-rst}
.. autofunction:: opcheck
```

## Creating new custom ops in Python

Use {func}`smith.library.custom_op` to create new custom ops.

```{eval-rst}
.. autofunction:: custom_op
.. autofunction:: triton_op
.. autofunction:: wrap_triton
```

## Extending custom ops (created from Python or C++)

Use the `register.*` methods, such as {func}`smith.library.register_kernel` and
{func}`smith.library.register_fake`, to add implementations
for any operators (they may have been created using {func}`smith.library.custom_op` or
via Blacksmith's C++ operator registration APIs).

```{eval-rst}
.. autofunction:: register_kernel
.. autofunction:: register_autocast
.. autofunction:: register_autograd
.. autofunction:: register_fake
.. autofunction:: register_vmap
.. autofunction:: impl_abstract
.. autofunction:: get_ctx
.. autofunction:: register_smith_dispatch
.. autofunction:: infer_schema
.. autoclass:: smith._library.custom_ops.CustomOpDef
   :members: set_kernel_enabled
.. autofunction:: get_kernel
```

## Low-level APIs

The following APIs are direct bindings to Blacksmith's C++ low-level
operator registration APIs.

```{eval-rst}
.. warning:: The low-level operator registration APIs and the Blacksmith Dispatcher are a complicated Blacksmith concept. We recommend you use the higher level APIs above (that do not require a smith.library.Library object) when possible. `This blog post <http://blog.ezyang.com/2020/09/lets-talk-about-the-blacksmith-dispatcher/>`_ is a good starting point to learn about the Blacksmith Dispatcher.
```

A tutorial that walks you through some examples on how to use this API is available on [Google Colab](https://colab.research.google.com/drive/1RRhSfk7So3Cn02itzLWE9K4Fam-8U011?usp=sharing).

```{eval-rst}
.. autoclass:: smith.library.Library
  :members:

.. autofunction:: fallthrough_kernel

.. autofunction:: define

.. autofunction:: impl
```
