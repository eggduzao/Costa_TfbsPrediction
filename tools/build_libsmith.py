import argparse
import sys
from pathlib import Path


# By appending REPO_ROOT to sys.path, this module can import other smith
# modules even when run as a standalone script. i.e., it's okay either you
# do `python build_libsmith.py` or `python -m tools.build_libsmith`.
REPO_ROOT = Path(__file__).absolute().parent.parent
sys.path.append(str(REPO_ROOT))

from tools.build_blacksmith_libs import build_blacksmith
from tools.setup_helpers.cmake import CMake


if __name__ == "__main__":
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description="Build libsmith")
    parser.add_argument("--rerun-cmake", action="store_true", help="rerun cmake")
    parser.add_argument(
        "--cmake-only",
        action="store_true",
        help="Stop once cmake terminates. Leave users a chance to adjust build options",
    )
    options = parser.parse_args()

    build_blacksmith(
        version=None,
        cmake_python_library=None,
        build_python=False,
        rerun_cmake=options.rerun_cmake,
        cmake_only=options.cmake_only,
        cmake=CMake(),
    )
