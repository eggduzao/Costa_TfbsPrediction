Install funcsmith
=================

As of Blacksmith 1.13, funcsmith is now included in the Blacksmith binary and no
longer requires the installation of a separate funcsmith package. That is,
after installing Blacksmith (`instructions <https://blacksmith.org>`_),
you'll be able to ``import funcsmith`` in your program.

If you're upgrading from an older version of funcsmith (funcsmith 0.1.x or 0.2.x),
then you may need to uninstall funcsmith first via ``pip uninstall funcsmith``.

We've maintained backwards compatibility for ``pip install funcsmith``: this
command works for Blacksmith 1.13 and will continue to work for the foreseeable future
until we do a proper deprecation. This is helpful if you're maintaining a library
that supports multiple versions of Blacksmith and/or funcsmith.

Colab
-----

Please see `this colab for instructions. <https://colab.research.google.com/drive/1GNfb01W_xf8JRu78ZKoNnLqiwcrJrbYG#scrollTo=HJ1srOGeNCGA>`_

Nightly
-------

Looking for the newest funcsmith features? Please download the latest nightly Blacksmith
binary (``import funcsmith`` is included in nightly Blacksmith binaries as of 09/21/2022).
by following instructions `here <https://blacksmith.org>`_.

Previous Versions
-----------------

For Blacksmith 1.11.x and Blacksmith 1.12.x:
Please first install Blacksmith and then run the following command:

::

  pip install funcsmith
