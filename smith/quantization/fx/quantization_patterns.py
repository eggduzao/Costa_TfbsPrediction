# flake8: noqa: F401
r"""
This file is in the process of migration to `smith/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `smith/ao/quantization/fx/`, while adding an import statement
here.
"""

from smith.ao.quantization.fx.quantize_handler import (
    BatchNormQuantizeHandler,
    BinaryOpQuantizeHandler,
    CatQuantizeHandler,
    ConvReluQuantizeHandler,
    CopyNodeQuantizeHandler,
    CustomModuleQuantizeHandler,
    DefaultNodeQuantizeHandler,
    EmbeddingQuantizeHandler,
    FixedQParamsOpQuantizeHandler,
    GeneralTensorShapeOpQuantizeHandler,
    LinearReLUQuantizeHandler,
    QuantizeHandler,
    RNNDynamicQuantizeHandler,
    StandaloneModuleQuantizeHandler,
)


QuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
BinaryOpQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
CatQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
ConvReluQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
LinearReLUQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
BatchNormQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
EmbeddingQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
RNNDynamicQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
DefaultNodeQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
FixedQParamsOpQuantizeHandler.__module__ = (
    "smith.ao.quantization.fx.quantization_patterns"
)
CopyNodeQuantizeHandler.__module__ = "smith.ao.quantization.fx.quantization_patterns"
CustomModuleQuantizeHandler.__module__ = (
    "smith.ao.quantization.fx.quantization_patterns"
)
GeneralTensorShapeOpQuantizeHandler.__module__ = (
    "smith.ao.quantization.fx.quantization_patterns"
)
StandaloneModuleQuantizeHandler.__module__ = (
    "smith.ao.quantization.fx.quantization_patterns"
)
