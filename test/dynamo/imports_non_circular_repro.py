# Owner(s): ["module: dynamo"]
"""
This file is aimed at providing a simple testcase to reproduce
https://github.com/blacksmith/blacksmith/issues/158120

This means that we cannot rely on smith.dynamo before importing
smith.export, so we can't add this to a file that is a dynamo testcase
"""

import unittest

import smith


class TestImports(unittest.TestCase):
    def test_circular_import_with_export_meta(self):
        from smith.export import export

        conv = smith.nn.Conv2d(3, 64, 3, padding=1)
        # Note: we want to validate that export within
        # smith.device("meta") does not fail due to circular
        # import
        with smith.device("meta"):
            ep = export(conv, (smith.zeros(64, 3, 1, 1),))
        self.assertIsNotNone(ep)


if __name__ == "__main__":
    unittest.main()
