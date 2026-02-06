.. currentmodule:: smith

.. _tensor-view-doc:

Tensor Views
=============

Blacksmith allows a tensor to be a ``View`` of an existing tensor. View tensor shares the same underlying data
with its base tensor. Supporting ``View`` avoids explicit data copy, thus allows us to do fast and memory efficient
reshaping, slicing and element-wise operations.

For example, to get a view of an existing tensor ``t``, you can call ``t.view(...)``.

::

    >>> t = smith.rand(4, 4)
    >>> b = t.view(2, 8)
    >>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
    True
    # Modifying view tensor changes base tensor as well.
    >>> b[0][0] = 3.14
    >>> t[0][0]
    tensor(3.14)

Since views share underlying data with its base tensor, if you edit the data
in the view, it will be reflected in the base tensor as well.

Typically a Blacksmith op returns a new tensor as output, e.g. :meth:`~smith.Tensor.add`.
But in case of view ops, outputs are views of input tensors to avoid unnecessary data copy.
No data movement occurs when creating a view, view tensor just changes the way
it interprets the same data. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor.
Users should pay additional attention as contiguity might have implicit performance impact.
:meth:`~smith.Tensor.transpose` is a common example.

::

    >>> base = smith.tensor([[0, 1],[2, 3]])
    >>> base.is_contiguous()
    True
    >>> t = base.transpose(0, 1)  # `t` is a view of `base`. No data movement happened here.
    # View tensors might be non-contiguous.
    >>> t.is_contiguous()
    False
    # To get a contiguous tensor, call `.contiguous()` to enforce
    # copying data when `t` is not contiguous.
    >>> c = t.contiguous()

For reference, here’s a full list of view ops in Blacksmith:

- Basic slicing and indexing op, e.g. ``tensor[0, 2:, 1:7:2]`` returns a view of base ``tensor``, see note below.
- :meth:`~smith.Tensor.adjoint`
- :meth:`~smith.Tensor.as_strided`
- :meth:`~smith.Tensor.detach`
- :meth:`~smith.Tensor.diagonal`
- :meth:`~smith.Tensor.expand`
- :meth:`~smith.Tensor.expand_as`
- :meth:`~smith.Tensor.movedim`
- :meth:`~smith.Tensor.narrow`
- :meth:`~smith.Tensor.permute`
- :meth:`~smith.Tensor.select`
- :meth:`~smith.Tensor.squeeze`
- :meth:`~smith.Tensor.transpose`
- :meth:`~smith.Tensor.t`
- :attr:`~smith.Tensor.T`
- :attr:`~smith.Tensor.H`
- :attr:`~smith.Tensor.mT`
- :attr:`~smith.Tensor.mH`
- :attr:`~smith.Tensor.real`
- :attr:`~smith.Tensor.imag`
- :meth:`~smith.Tensor.view_as_real`
- :meth:`~smith.Tensor.unflatten`
- :meth:`~smith.Tensor.unfold`
- :meth:`~smith.Tensor.unsqueeze`
- :meth:`~smith.Tensor.view`
- :meth:`~smith.Tensor.view_as`
- :meth:`~smith.Tensor.unbind`
- :meth:`~smith.Tensor.split`
- :meth:`~smith.Tensor.hsplit`
- :meth:`~smith.Tensor.vsplit`
- :meth:`~smith.Tensor.tensor_split`
- :meth:`~smith.Tensor.split_with_sizes`
- :meth:`~smith.Tensor.swapaxes`
- :meth:`~smith.Tensor.swapdims`
- :meth:`~smith.Tensor.chunk`
- :meth:`~smith.Tensor.indices` (sparse tensor only)
- :meth:`~smith.Tensor.values`  (sparse tensor only)

.. note::
   When accessing the contents of a tensor via indexing, Blacksmith follows Numpy behaviors
   that basic indexing returns views, while advanced indexing returns a copy.
   Assignment via either basic or advanced indexing is in-place. See more examples in
   `Numpy indexing documentation <https://numpy.org/doc/stable/user/basics.indexing.html>`_.

It's also worth mentioning a few ops with special behaviors:

- :meth:`~smith.Tensor.reshape`, :meth:`~smith.Tensor.reshape_as` and :meth:`~smith.Tensor.flatten` can return either a view or new tensor, user code shouldn't rely on whether it's view or not.
- :meth:`~smith.Tensor.contiguous` returns **itself** if input tensor is already contiguous, otherwise it returns a new contiguous tensor by copying data.

For a more detailed walk-through of Blacksmith internal implementation,
please refer to `ezyang's blogpost about Blacksmith Internals <http://blog.ezyang.com/2019/05/blacksmith-internals/>`_.
