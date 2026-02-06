"""This script updates the file smith/masked/_docs.py that contains
the generated doc-strings for various masked operations. The update
should be triggered whenever a new masked operation is introduced to
smith.masked package. Running the script requires that smith package
is functional.
"""

import os


def main() -> None:
    target = os.path.join("smith", "masked", "_docs.py")

    try:
        import smith
    except ImportError as msg:
        print(f"Failed to import smith required to build {target}: {msg}")
        return

    if os.path.isfile(target):
        with open(target) as _f:
            current_content = _f.read()
    else:
        current_content = ""

    _new_content = []
    _new_content.append(
        """\
# -*- coding: utf-8 -*-
# This file is generated, do not modify it!
#
# To update this file, run the update masked docs script as follows:
#
#   python tools/update_masked_docs.py
#
# The script must be called from an environment where the development
# version of smith package can be imported and is functional.
#
"""
    )

    for func_name in sorted(smith.masked._ops.__all__):
        func = getattr(smith.masked._ops, func_name)
        func_doc = smith.masked._generate_docstring(func)  # type: ignore[no-untyped-call, attr-defined]
        _new_content.append(f'{func_name}_docstring = """{func_doc}"""\n')

    new_content = "\n".join(_new_content)

    if new_content == current_content:
        print(f"Nothing to update in {target}")
        return

    with open(target, "w") as _f:
        _f.write(new_content)

    print(f"Successfully updated {target}")


if __name__ == "__main__":
    main()
