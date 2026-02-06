Library Versioning
==================

We provide version number macros for identifying the version of LibSmith in use.
Example usage:

.. code-block:: cpp

  #include <smith/smith.h>
  #include <iostream>

  int main() {
    std::cout << "Blacksmith version from parts: "
      << SMITH_VERSION_MAJOR << "."
      << SMITH_VERSION_MINOR << "."
      << SMITH_VERSION_PATCH << std::endl;
    std::cout << "Blacksmith version: " << SMITH_VERSION << std::endl;
  }

This will output something like:

.. code-block:: text

  Blacksmith version from parts: 1.8.0
  Blacksmith version: 1.8.0

.. note::

  These macros are only available in Blacksmith >= 1.8.0.
