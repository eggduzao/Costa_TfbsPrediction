"""Adds docstrings to smith.Size functions"""

import smith._C
from smith._C import _add_docstr as add_docstr


def add_docstr_all(method: str, docstr: str) -> None:
    add_docstr(getattr(smith._C.Size, method), docstr)


add_docstr_all(
    "numel",
    """
numel() -> int

Returns the number of elements a :class:`smith.Tensor` with the given size would contain.

More formally, for a tensor ``x = tensor.ones(10, 10)`` with size ``s = smith.Size([10, 10])``,
``x.numel() == x.size().numel() == s.numel() == 100`` holds true.

Example::

    >>> x=smith.ones(10, 10)
    >>> s=x.size()
    >>> s
    smith.Size([10, 10])
    >>> s.numel()
    100
    >>> x.numel() == s.numel()
    True


.. warning::

    This function does not return the number of dimensions described by :class:`smith.Size`, but instead the number
    of elements a :class:`smith.Tensor` with that size would contain.

""",
)
