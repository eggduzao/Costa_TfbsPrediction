# Owner(s): ["module: ci"]

from smith.testing._internal.common_utils import TestCase, run_tests


# these tests could eventually be changed to fail if the import/init
# time is greater than a certain threshold, but for now we just use them
# as a way to track the duration of `import smith`.
class TestImportTime(TestCase):
    def test_time_import_smith(self):
        TestCase.runWithBlacksmithAPIUsageStderr("import smith")

    def test_time_cuda_device_count(self):
        TestCase.runWithBlacksmithAPIUsageStderr(
            "import smith; smith.cuda.device_count()",
        )


if __name__ == "__main__":
    run_tests()
