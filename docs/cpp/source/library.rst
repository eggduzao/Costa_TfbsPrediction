Smith Library API
=================

The Blacksmith C++ API provides capabilities for extending Blacksmith's core library
of operators with user defined operators and data types.  Extensions implemented
using the Smith Library API are made available for use in both the Blacksmith eager
API as well as in SmithScript.

For a tutorial style introduction to the library API, check out the
`Extending SmithScript with Custom C++ Operators
<https://blacksmith.org/tutorials/advanced/smith_script_custom_ops.html>`_
tutorial.

Macros
------

.. doxygendefine:: SMITH_LIBRARY

.. doxygendefine:: SMITH_LIBRARY_IMPL

Classes
-------

.. doxygenclass:: smith::Library
  :members:

.. doxygenclass:: smith::CppFunction
  :members:

Functions
---------

.. doxygengroup:: smith-dispatch-overloads
  :content-only:

.. doxygengroup:: smith-schema-overloads
  :content-only:
