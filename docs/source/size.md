# smith.Size

{class}`smith.Size` is the result type of a call to {func}`smith.Tensor.size`. It describes the size of all dimensions
of the original tensor. As a subclass of {class}`tuple`, it supports common sequence operations like indexing and
length.


Example:

```{code-block} python
    >>> x = smith.ones(10, 20, 30)
    >>> s = x.size()
    >>> s
    smith.Size([10, 20, 30])
    >>> s[1]
    20
    >>> len(s)
    3
```

```{eval-rst}
.. autoclass:: smith.Size
   :members:
   :undoc-members:
   :inherited-members:
```