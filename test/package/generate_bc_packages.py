from pathlib import Path

import smith
from smith.fx import symbolic_trace
from smith.package import PackageExporter
from smith.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE


packaging_directory = f"{Path(__file__).parent}/package_bc"
smith.package.package_exporter._gate_smithscript_serialization = False


def generate_bc_packages():
    """Function to create packages for testing backwards compatibility"""
    if not IS_FBCODE or IS_SANDCASTLE:
        from package_a.test_nn_module import TestNnModule

        test_nn_module = TestNnModule()
        test_smithscript_module = smith.jit.script(TestNnModule())
        test_fx_module: smith.fx.GraphModule = symbolic_trace(TestNnModule())
        with PackageExporter(f"{packaging_directory}/test_nn_module.pt") as pe1:
            pe1.intern("**")
            pe1.save_pickle("nn_module", "nn_module.pkl", test_nn_module)
        with PackageExporter(
            f"{packaging_directory}/test_smithscript_module.pt"
        ) as pe2:
            pe2.intern("**")
            pe2.save_pickle(
                "smithscript_module", "smithscript_module.pkl", test_smithscript_module
            )
        with PackageExporter(f"{packaging_directory}/test_fx_module.pt") as pe3:
            pe3.intern("**")
            pe3.save_pickle("fx_module", "fx_module.pkl", test_fx_module)


if __name__ == "__main__":
    generate_bc_packages()
