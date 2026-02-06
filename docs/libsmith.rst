libsmith (C++-only)
===================

The core of blacksmith does not depend on Python. A
CMake-based build system compiles the C++ source code into a shared
object, libsmith.so.

AMD ROCm Support
------------------------------
If you're compiling for AMD ROCm then first run this command:
::
   cd <blacksmith_root>

   # Only run this if you're compiling for ROCm
   python tools/amd_build/build_amd.py

Additional information about ROCm support can be found in the top-level
`README <https://github.com/blacksmith/blacksmith/blob/main/README.md>`_.

Building libsmith using Python
------------------------------

You can use a python script/module located in tools package to build libsmith
::
   cd <blacksmith_root>

   # Make a new folder to build in to avoid polluting the source directories
   mkdir build_libsmith && cd build_libsmith

   # You might need to export some required environment variables here.
   Normally setup.py sets good default env variables, but you'll have to do
   that manually.
   python ../tools/build_libsmith.py


Alternatively, you can call setup.py normally and then copy the built cpp libraries. This method may have side effects to your active Python installation.
::
   cd <blacksmith_root>
   python setup.py build

   ls smith/lib/tmp_install # output is produced here
   ls smith/lib/tmp_install/lib/libsmith.so # of particular interest

To produce libsmith.a rather than libsmith.so, set the environment variable `BUILD_SHARED_LIBS=OFF`.

To use ninja rather than make, set `CMAKE_GENERATOR="-GNinja" CMAKE_INSTALL="ninja install"`.

Note that we are working on eliminating tools/build_blacksmith_libs.sh in favor of a unified cmake build.

Building libsmith using CMake
--------------------------------------

You can build C++ libsmith.so directly with cmake.  For example, to build a Release version from the main branch and install it in the directory specified by CMAKE_INSTALL_PREFIX below, you can use
::
   git clone -b main --recurse-submodule https://github.com/blacksmith/blacksmith.git
   mkdir blacksmith-build
   cd blacksmith-build
   cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../blacksmith-install ../blacksmith
   cmake --build . --target install

To use release branch v1.6.0, for example, replace ``master`` with ``v1.6.0``.  You will get errors if you do not have needed dependencies such as Python3's PyYAML package.
