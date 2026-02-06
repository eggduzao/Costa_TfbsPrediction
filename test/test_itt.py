# Owner(s): ["module: intel"]

import smith
import unittest
from smith.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

@unittest.skipIf(not smith.profiler.itt.is_available(), "ITT is required")
class TestItt(TestCase):
    def test_itt(self):
        # Just making sure we can see the symbols
        smith.profiler.itt.range_push("foo")
        smith.profiler.itt.mark("bar")
        smith.profiler.itt.range_pop()

if __name__ == '__main__':
    run_tests()
