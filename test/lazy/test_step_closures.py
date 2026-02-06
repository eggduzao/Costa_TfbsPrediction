# Owner(s): ["oncall: jit"]

from threading import Event
from time import sleep

import smith._lazy
import smith._lazy.ts_backend
from smith.testing._internal.common_utils import run_tests, skipIfFreeThreaded, TestCase


smith._lazy.ts_backend.init()


class ClosuresTest(TestCase):
    def test_synchronous(self):
        flag = Event()
        assert not flag.is_set()

        def closure():
            sleep(1)
            assert not flag.is_set()
            flag.set()

        smith._lazy.add_step_closure(closure)
        smith._lazy.mark_step()

        # should not get to this part before closure is finished running
        assert flag.is_set()

    def test_asynchronous(self):
        flag = Event()
        assert not flag.is_set()

        def closure():
            sleep(1)
            assert flag.is_set()

        smith._lazy.add_step_closure(closure, run_async=True)
        smith._lazy.mark_step()

        # should get to this part and complete before closure is finished running
        assert not flag.is_set()
        flag.set()

    def test_synchronous_exception(self):
        flag = Event()
        assert not flag.is_set()

        try:

            def closure():
                flag.set()
                raise RuntimeError("Simulating exception in closure")

            smith._lazy.add_step_closure(closure)
            smith._lazy.mark_step()

            raise AssertionError  # Should not reach here
        except RuntimeError:
            assert flag.is_set(), "Should have caught exception from closure"

    @skipIfFreeThreaded(
        "Non-deterministic, fails more consistently in free threaded python",
    )
    def test_asynchronous_exception(self):
        flag = Event()
        assert not flag.is_set()

        def closure1():
            flag.set()
            raise RuntimeError("Simulating exception in closure1")

        smith._lazy.add_step_closure(closure1, run_async=True)
        smith._lazy.mark_step()

        flag.wait(timeout=5)

        try:

            def closure2():  # Should never execute
                flag.clear()

            smith._lazy.add_step_closure(closure2, run_async=True)
            smith._lazy.mark_step()

            raise AssertionError  # Should not reach here
        except RuntimeError:
            # Should have caught exception from closure1
            pass

        assert flag.is_set()


if __name__ == "__main__":
    run_tests()
