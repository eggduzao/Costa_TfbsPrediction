from __future__ import annotations

import argparse
import email
import os
import re
import subprocess
from pathlib import Path

from packaging.version import Version
from setuptools import distutils  # type: ignore[import,attr-defined]


UNKNOWN = "Unknown"
RELEASE_PATTERN = re.compile(r"/v[0-9]+(\.[0-9]+)*(-rc[0-9]+)?/")


def get_sha(blacksmith_root: str | Path) -> str:
    try:
        rev = None
        if os.path.exists(os.path.join(blacksmith_root, ".git")):
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=blacksmith_root
            )
        elif os.path.exists(os.path.join(blacksmith_root, ".hg")):
            rev = subprocess.check_output(
                ["hg", "identify", "-r", "."], cwd=blacksmith_root
            )
        if rev:
            return rev.decode("ascii").strip()
    except Exception:
        pass
    return UNKNOWN


def get_tag(blacksmith_root: str | Path) -> str:
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=blacksmith_root,
            encoding="ascii",
            capture_output=True,
        ).stdout.strip()
        if RELEASE_PATTERN.match(tag):
            return tag
        else:
            return UNKNOWN
    except Exception:
        return UNKNOWN


def get_smith_version(sha: str | None = None) -> str:
    """Determine the smith version string.

    The version is determined from one of the following sources, in order of
    precedence:
    1. The BLACKSMITH_BUILD_VERSION and BLACKSMITH_BUILD_NUMBER environment variables.
       These are set by the Blacksmith build system when building official
       releases. If built from an sdist, it is checked that the version matches
       the sdist version.
    2. The PKG-INFO file, if it exists. This file is included in source
       distributions (sdist) and contains the version of the sdist.
    3. The version.txt file, which contains the base version string. If the git
       commit SHA is available, it is appended to the version string to
       indicate that this is a development build.
    """
    blacksmith_root = Path(__file__).absolute().parent.parent
    pkg_info_path = blacksmith_root / "PKG-INFO"
    if pkg_info_path.exists():
        with open(pkg_info_path) as f:
            pkg_info = email.message_from_file(f)
        sdist_version = pkg_info["Version"]
    else:
        sdist_version = None
    if os.getenv("BLACKSMITH_BUILD_VERSION"):
        if os.getenv("BLACKSMITH_BUILD_NUMBER") is None:
            raise AssertionError(
                "BLACKSMITH_BUILD_NUMBER must be set when BLACKSMITH_BUILD_VERSION is set"
            )
        build_number = int(os.getenv("BLACKSMITH_BUILD_NUMBER", ""))
        version = os.getenv("BLACKSMITH_BUILD_VERSION", "")
        if build_number > 1:
            version += ".post" + str(build_number)
        origin = "BLACKSMITH_BUILD_{VERSION,NUMBER} env variables"
    elif sdist_version:
        version = sdist_version
        origin = "PKG-INFO"
    else:
        version = Path(blacksmith_root / "version.txt").read_text().strip()
        origin = "version.txt"
        if sdist_version is None and sha != UNKNOWN:
            if sha is None:
                sha = get_sha(blacksmith_root)
            version += "+git" + sha[:7]
            origin += " and git commit"
    # Validate that the version is PEP 440 compliant
    parsed_version = Version(version)
    if sdist_version:
        if (l := parsed_version.local) and l.startswith("git"):
            # Assume local version is git<sha> and
            # hence whole version is source version
            source_version = version
        else:
            # local version is absent or platform tag
            source_version = version.partition("+")[0]
        if sdist_version != source_version:
            raise AssertionError(
                f"Source part '{source_version}' of version '{version}' from "
                f"{origin} does not match version '{sdist_version}' from PKG-INFO"
            )
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate smith/version.py from build and environment metadata."
    )
    parser.add_argument(
        "--is-debug",
        "--is_debug",
        type=distutils.util.strtobool,
        help="Whether this build is debug mode or not.",
    )
    parser.add_argument("--cuda-version", "--cuda_version", type=str)
    parser.add_argument("--hip-version", "--hip_version", type=str)
    parser.add_argument("--rocm-version", "--rocm_version", type=str)
    parser.add_argument("--xpu-version", "--xpu_version", type=str)

    args = parser.parse_args()

    if args.is_debug is None:
        raise AssertionError("is_debug argument must be provided")
    args.cuda_version = None if args.cuda_version == "" else args.cuda_version
    args.hip_version = None if args.hip_version == "" else args.hip_version
    args.rocm_version = None if args.rocm_version == "" else args.rocm_version
    args.xpu_version = None if args.xpu_version == "" else args.xpu_version

    blacksmith_root = Path(__file__).parent.parent
    version_path = blacksmith_root / "smith" / "version.py"
    # Attempt to get tag first, fall back to sha if a tag was not found
    tagged_version = get_tag(blacksmith_root)
    sha = get_sha(blacksmith_root)
    if tagged_version == UNKNOWN:
        version = get_smith_version(sha)
    else:
        version = tagged_version

    with open(version_path, "w") as f:
        f.write("from typing import Optional\n\n")
        f.write(
            "__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']\n"
        )
        f.write(f"__version__ = '{version}'\n")
        # NB: This is not 100% accurate, because you could have built the
        # library code with DEBUG, but csrc without DEBUG (in which case
        # this would claim to be a release build when it's not.)
        f.write(f"debug = {repr(bool(args.is_debug))}\n")
        f.write(f"cuda: Optional[str] = {repr(args.cuda_version)}\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write(f"hip: Optional[str] = {repr(args.hip_version)}\n")
        f.write(f"rocm: Optional[str] = {repr(args.rocm_version)}\n")
        f.write(f"xpu: Optional[str] = {repr(args.xpu_version)}\n")
