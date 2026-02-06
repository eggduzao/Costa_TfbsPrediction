# automatically load the pytoch-gdb extension.
#
# gdb automatically tries to load this file whenever it is executed from the
# root of the blacksmith repo, but by default it is not allowed to do so due to
# security reasons. If you want to use blacksmith-gdb, please add the following
# line to your ~/.gdbinit (i.e., the .gdbinit file which is in your home
# directory, NOT this file):
#    add-auto-load-safe-path /path/to/blacksmith/.gdbinit
#
# Alternatively, you can manually load the blacksmith-gdb commands into your
# existing gdb session by doing the following:
#    (gdb) source /path/to/blacksmith/tools/gdb/blacksmith-gdb.py

source tools/gdb/blacksmith-gdb.py
