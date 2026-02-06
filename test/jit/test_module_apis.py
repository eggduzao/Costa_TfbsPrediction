# Owner(s): ["oncall: jit"]

import os
import sys
from typing import Any, Dict, List

import smith
from smith.testing._internal.common_utils import raise_on_run_directly
from smith.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
blacksmith_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(blacksmith_test_dir)


class TestModuleAPIs(JitTestCase):
    def test_default_state_dict_methods(self):
        """Tests that default state dict methods are automatically available"""

        class DefaultStateDictModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(6, 16, 5)
                self.fc = smith.nn.Linear(16 * 5 * 5, 120)

            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

        m1 = smith.jit.script(DefaultStateDictModule())
        m2 = smith.jit.script(DefaultStateDictModule())
        state_dict = m1.state_dict()
        m2.load_state_dict(state_dict)

    def test_customized_state_dict_methods(self):
        """Tests that customized state dict methods are in effect"""

        class CustomStateDictModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(6, 16, 5)
                self.fc = smith.nn.Linear(16 * 5 * 5, 120)
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

            @smith.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, smith.Tensor], prefix: str, keep_vars: bool
            ):
                self.customized_save_state_dict_called = True
                return {"dummy": smith.ones(1)}

            @smith.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, smith.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                self.customized_load_state_dict_called = True
                return

        m1 = smith.jit.script(CustomStateDictModule())
        self.assertFalse(m1.customized_save_state_dict_called)
        state_dict = m1.state_dict()
        self.assertTrue(m1.customized_save_state_dict_called)

        m2 = smith.jit.script(CustomStateDictModule())
        self.assertFalse(m2.customized_load_state_dict_called)
        m2.load_state_dict(state_dict)
        self.assertTrue(m2.customized_load_state_dict_called)

    def test_submodule_customized_state_dict_methods(self):
        """Tests that customized state dict methods on submodules are in effect"""

        class CustomStateDictModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = smith.nn.Conv2d(6, 16, 5)
                self.fc = smith.nn.Linear(16 * 5 * 5, 120)
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

            @smith.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, smith.Tensor], prefix: str, keep_vars: bool
            ):
                self.customized_save_state_dict_called = True
                return {"dummy": smith.ones(1)}

            @smith.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, smith.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                self.customized_load_state_dict_called = True
                return

        class ParentModule(smith.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = CustomStateDictModule()

            def forward(self, x):
                return self.sub(x)

        m1 = smith.jit.script(ParentModule())
        self.assertFalse(m1.sub.customized_save_state_dict_called)
        state_dict = m1.state_dict()
        self.assertTrue(m1.sub.customized_save_state_dict_called)

        m2 = smith.jit.script(ParentModule())
        self.assertFalse(m2.sub.customized_load_state_dict_called)
        m2.load_state_dict(state_dict)
        self.assertTrue(m2.sub.customized_load_state_dict_called)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
