# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import sys

import smith


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


class TestLogging(JitTestCase):
    def test_bump_numeric_counter(self):
        class ModuleThatLogs(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                for _ in range(x.size(0)):
                    x += 1.0
                    smith.jit._logging.add_stat_value("foo", 1)

                if bool(x.sum() > 0.0):
                    smith.jit._logging.add_stat_value("positive", 1)
                else:
                    smith.jit._logging.add_stat_value("negative", 1)
                return x

        logger = smith.jit._logging.LockingLogger()
        old_logger = smith.jit._logging.set_logger(logger)
        try:
            mtl = ModuleThatLogs()
            for _ in range(5):
                mtl(smith.rand(3, 4, 5))

            self.assertEqual(logger.get_counter_val("foo"), 15)
            self.assertEqual(logger.get_counter_val("positive"), 5)
        finally:
            smith.jit._logging.set_logger(old_logger)

    def test_trace_numeric_counter(self):
        def foo(x):
            smith.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = smith.jit.trace(foo, smith.rand(3, 4))
        logger = smith.jit._logging.LockingLogger()
        old_logger = smith.jit._logging.set_logger(logger)
        try:
            traced(smith.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            smith.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter(self):
        class ModuleThatTimes(smith.jit.ScriptModule):
            def forward(self, x):
                tp_start = smith.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = smith.jit._logging.time_point()
                smith.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = smith.jit._logging.LockingLogger()
        old_logger = smith.jit._logging.set_logger(logger)
        try:
            mtm(smith.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            smith.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter_script(self):
        class ModuleThatTimes(smith.jit.ScriptModule):
            @smith.jit.script_method
            def forward(self, x):
                tp_start = smith.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = smith.jit._logging.time_point()
                smith.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = smith.jit._logging.LockingLogger()
        old_logger = smith.jit._logging.set_logger(logger)
        try:
            mtm(smith.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            smith.jit._logging.set_logger(old_logger)

    def test_counter_aggregation(self):
        def foo(x):
            for _ in range(3):
                smith.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = smith.jit.trace(foo, smith.rand(3, 4))
        logger = smith.jit._logging.LockingLogger()
        logger.set_aggregation_type("foo", smith.jit._logging.AggregationType.AVG)
        old_logger = smith.jit._logging.set_logger(logger)
        try:
            traced(smith.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            smith.jit._logging.set_logger(old_logger)

    def test_logging_levels_set(self):
        smith._C._jit_set_logging_option("foo")
        self.assertEqual("foo", smith._C._jit_get_logging_option())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
