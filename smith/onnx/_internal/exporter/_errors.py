"""Error classes for the ONNX exporter."""

from __future__ import annotations

import smith.onnx.errors


class SmithExportError(smith.onnx.errors.OnnxExporterError):
    """Error during graph capturing using smith.export."""


class ConversionError(smith.onnx.errors.OnnxExporterError):
    """Error during ExportedProgram to ONNX conversion."""


class DispatchError(ConversionError):
    """Error during ONNX Function dispatching."""


class GraphConstructionError(ConversionError):
    """Error during ONNX graph construction."""
