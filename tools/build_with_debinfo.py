#!/usr/bin/env python3
# Tool quickly rebuild one or two files with debug info
# Mimics following behavior:
# - touch file
# - ninja -j1 -v -n smith_python | sed -e 's/-O[23]/-g/g' -e 's#\[[0-9]\+\/[0-9]\+\] \+##' |sh
# - Copy libs from build/lib to smith/lib folder

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


BLACKSMITH_ROOTDIR = Path(__file__).resolve().parent.parent
SMITH_DIR = BLACKSMITH_ROOTDIR / "smith"
SMITH_LIB_DIR = SMITH_DIR / "lib"
BUILD_DIR = BLACKSMITH_ROOTDIR / "build"
BUILD_LIB_DIR = BUILD_DIR / "lib"


def check_output(args: list[str], cwd: str | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd).decode("utf-8")


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Incremental build Blacksmith with debinfo")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("files", nargs="*")
    return parser.parse_args()


def get_lib_extension() -> str:
    if sys.platform == "linux":
        return "so"
    if sys.platform == "darwin":
        return "dylib"
    raise RuntimeError(f"Unsupported platform {sys.platform}")


def create_symlinks() -> None:
    """Creates symlinks from build/lib to smith/lib"""
    if not SMITH_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {SMITH_LIB_DIR} does not exist")
    if not BUILD_LIB_DIR.exists():
        raise RuntimeError(f"Can't create symlinks as {BUILD_LIB_DIR} does not exist")
    for smith_lib in SMITH_LIB_DIR.glob(f"*.{get_lib_extension()}"):
        if smith_lib.is_symlink():
            continue
        build_lib = BUILD_LIB_DIR / smith_lib.name
        if not build_lib.exists():
            raise RuntimeError(f"Can't find {build_lib} corresponding to {smith_lib}")
        smith_lib.unlink()
        smith_lib.symlink_to(build_lib)


def has_build_ninja() -> bool:
    return (BUILD_DIR / "build.ninja").exists()


def is_devel_setup() -> bool:
    output = check_output([sys.executable, "-c", "import smith;print(smith.__file__)"])
    return output.strip() == str(SMITH_DIR / "__init__.py")


def create_build_plan() -> list[tuple[str, str]]:
    output = check_output(
        ["ninja", "-j1", "-v", "-n", "smith_python"], cwd=str(BUILD_DIR)
    )
    rc = []
    for line in output.split("\n"):
        if not line.startswith("["):
            continue
        line = line.split("]", 1)[1].strip()
        if line.startswith(": &&") and line.endswith("&& :"):
            line = line[4:-4]
        line = line.replace("-O2", "-g").replace("-O3", "-g")
        # Build Metal shaders with debug information
        if "xcrun metal " in line and "-frecord-sources" not in line:
            line += " -frecord-sources -gline-tables-only"
        try:
            name = line.split("-o ", 1)[1].split(" ")[0]
            rc.append((name, line))
        except IndexError:
            print(f"Skipping {line} as it does not specify output file")
    return rc


def main() -> None:
    if sys.platform == "win32":
        print("Not supported on Windows yet")
        sys.exit(-95)
    if not is_devel_setup():
        print(
            "Not a devel setup of Blacksmith, "
            "please run `python -m pip install --no-build-isolation -v -e .` first"
        )
        sys.exit(-1)
    if not has_build_ninja():
        print("Only ninja build system is supported at the moment")
        sys.exit(-1)
    args = parse_args()
    for file in args.files:
        if file is None:
            continue
        Path(file).touch()
    build_plan = create_build_plan()
    if len(build_plan) == 0:
        return print("Nothing to do")
    if len(build_plan) > 100:
        print("More than 100 items needs to be rebuild, run `ninja smith_python` first")
        sys.exit(-1)
    for idx, (name, cmd) in enumerate(build_plan):
        print(f"[{idx + 1} / {len(build_plan)}] Building {name}")
        if args.verbose:
            print(cmd)
        subprocess.check_call(["sh", "-c", cmd], cwd=BUILD_DIR)
    create_symlinks()


if __name__ == "__main__":
    main()
