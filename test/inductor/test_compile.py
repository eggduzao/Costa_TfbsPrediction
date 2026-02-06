# Owner(s): ["module: inductor"]
import os
import shlex
import subprocess
import sys
from unittest import mock

import smith
from smith import _dynamo as dynamo, _inductor as inductor
from smith._inductor.codecache import write
from smith._inductor.cpp_builder import CppBuilder, CppOptions
from smith._inductor.test_case import run_tests, TestCase
from smith._inductor.utils import gen_gm_and_inputs
from smith.fx import symbolic_trace
from smith.fx.experimental.proxy_tensor import make_fx
from smith.testing._internal.inductor_utils import HAS_CPU


_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"


def safe_command_output(cmd, timeout=30):
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        ).strip()
    except subprocess.CalledProcessError as e:
        return f"run failed（error code {e.returncode}）: {e.output.strip()}"
    except subprocess.TimeoutExpired:
        return "runt timeout"


class MyModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = smith.nn.Linear(10, 10)
        self.b = smith.nn.Linear(10, 10)
        self.relu = smith.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.a(x))
        x = smith.sigmoid(self.b(x))
        return x


class MyModule2(MyModule):
    def forward(self, x):  # takes a dict of list
        a, b = x["key"]
        return {"result": super().forward(a) + b}


class MyModule3(MyModule):
    def forward(self, x):
        return (super().forward(x),)


class TestStandaloneInductor(TestCase):
    """
    These test check that you can call SmithInductor directly without
    going through SmithDynamo.
    """

    def test_inductor_via_fx(self):
        mod = MyModule3().eval()
        inp = smith.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_tensor_return(self):
        mod = MyModule().eval()
        inp = smith.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_dict_input(self):
        mod = MyModule2().eval()
        inp = {"key": [smith.randn(10), smith.randn(10)]}
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_make_fx(self):
        mod = MyModule().eval()
        inp = smith.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(make_fx(mod)(inp), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_bare_module(self):
        mod = MyModule3().eval()
        inp = smith.randn(10)
        correct = mod(inp)
        # no FX graph at all (mod must return list/tuple in this case)
        mod_opt = inductor.compile(mod, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export1(self):
        mod = MyModule3().eval()
        inp = smith.randn(10)
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp, aten_graph=True)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export2(self):
        mod = MyModule2().eval()
        inp = {"key": [smith.randn(10), smith.randn(10)]}
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_op_with_multiple_outputs(self):
        x1 = smith.randn((2, 512, 128))
        x2 = [128]
        x3 = smith.randn(128)
        x4 = smith.randn((128,))
        x5 = 1e-6
        mod, inp = gen_gm_and_inputs(
            smith.ops.aten.native_layer_norm.default, (x1, x2, x3, x4, x5), {}
        )
        mod_opt = inductor.compile(mod, inp)
        self.assertEqual(mod(*inp), mod_opt(*inp))

    @mock.patch.dict(os.environ, {"SMITHINDUCTOR_DEBUG_COMPILE": "1"})
    def test_inductor_generate_debug_compile(self):
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_compile",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug compile.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)

    @mock.patch.dict(os.environ, {"SMITHINDUCTOR_DEBUG_SYMBOL": "1"})
    def test_inductor_generate_debug_symbol(self):
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_symbol",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug symbol.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
