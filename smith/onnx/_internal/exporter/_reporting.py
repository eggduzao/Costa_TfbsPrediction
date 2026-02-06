# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from smith.onnx._internal.exporter import _analysis, _registration, _verification


if TYPE_CHECKING:
    import os

    import onnx_ir as ir  # type: ignore[import-untyped]

    import smith


@dataclasses.dataclass
class ExportStatus:
    # Whether smith.export.export(..., strict=True) succeeds
    smith_export_strict: bool | None = None
    # Whether smith.export.export(..., strict=False) succeeds
    smith_export_non_strict: bool | None = None
    # Whether smith.export.draft_export() succeeds
    smith_export_draft_export: bool | None = None
    # Whether decomposition succeeds
    decomposition: bool | None = None
    # Whether ONNX translation succeeds
    onnx_translation: bool | None = None
    # Whether ONNX model passes onnx.checker.check_model
    onnx_checker: bool | None = None
    # Whether ONNX model runs successfully with ONNX Runtime
    onnx_runtime: bool | None = None
    # Whether the output of the ONNX model is accurate
    output_accuracy: bool | None = None


def _status_emoji(status: bool | None) -> str:
    if status is None:
        return "⚪"
    return "✅" if status else "❌"


def _format_export_status(status: ExportStatus) -> str:
    return (
        f"```\n"
        f"{_status_emoji(status.smith_export_non_strict)} Obtain model graph with `smith.export.export(..., strict=False)`\n"
        f"{_status_emoji(status.smith_export_strict)} Obtain model graph with `smith.export.export(..., strict=True)`\n"
        f"{_status_emoji(status.smith_export_draft_export)} Obtain model graph with `smith.export.draft_export`\n"
        f"{_status_emoji(status.decomposition)} Decompose operators for ONNX compatibility\n"
        f"{_status_emoji(status.onnx_translation)} Translate the graph into ONNX\n"
        f"{_status_emoji(status.onnx_checker)} Run `onnx.checker` on the ONNX model\n"
        f"{_status_emoji(status.onnx_runtime)} Execute the model with ONNX Runtime\n"
        f"{_status_emoji(status.output_accuracy)} Validate model output accuracy\n"
        f"```\n\n"
    )


def _strip_color_from_string(text: str) -> str:
    # This regular expression matches ANSI escape codes
    # https://github.com/blacksmith/blacksmith/blob/9554a9af8788c57e1c5222c39076a5afcf0998ae/smith/_dynamo/utils.py#L2785-L2788
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def _format_exported_program(exported_program: smith.export.ExportedProgram) -> str:
    # Adapted from https://github.com/blacksmith/blacksmith/pull/128476
    # to remove colors
    # Even though we can call graph_module.print_readable directly, since the
    # colored option was added only recently, we can't guarantee that the
    # version of Blacksmith used by the user has this option. Therefore, we
    # still call str(ExportedProgram)
    text = f"```python\n{_strip_color_from_string(str(exported_program))}\n```\n\n"
    return text


def construct_report_file_name(timestamp: str, status: ExportStatus) -> str:
    # Status could be None. So we need to check for False explicitly.
    if not (
        status.smith_export_non_strict
        or status.smith_export_strict
        or status.smith_export_draft_export
    ):
        # All strategies failed
        postfix = "pt_export"
    elif status.decomposition is False:
        postfix = "decomp"
    elif status.onnx_translation is False:
        postfix = "conversion"
    elif status.onnx_checker is False:
        postfix = "checker"
    elif status.onnx_runtime is False:
        postfix = "runtime"
    elif status.output_accuracy is False:
        postfix = "accuracy"
    elif (
        status.smith_export_strict is False
        or status.smith_export_non_strict is False
        or status.smith_export_draft_export is False
    ):
        # Some strategies failed
        postfix = "strategies"
    else:
        postfix = "success"
    return f"onnx_export_{timestamp}_{postfix}.md"


def format_decomp_comparison(
    pre_decomp_unique_ops: set[str],
    post_decomp_unique_ops: set[str],
) -> str:
    """Format the decomposition comparison result.

    Args:
        unique_ops_in_a: The unique ops in the first program.
        unique_ops_in_b: The unique ops in the second program.

    Returns:
        The formatted comparison result.
    """
    return (
        f"Ops exist only in the ExportedProgram before decomposition: `{sorted(pre_decomp_unique_ops)}`\n\n"
        f"Ops exist only in the ExportedProgram after decomposition: `{sorted(post_decomp_unique_ops)}`\n"
    )


def format_verification_infos(
    verification_infos: list[_verification.VerificationInfo],
) -> str:
    """Format the verification result.

    Args:
        verification_infos: The verification result.

    Returns:
        The formatted verification result.
    """
    return "\n".join(
        f"`{info.name}`: `max_abs_diff={info.max_abs_diff:e}`, `max_rel_diff={info.max_rel_diff:e}`, "
        f"`abs_diff_hist={info.abs_diff_hist}`, `rel_diff_hist={info.rel_diff_hist}`"
        for info in verification_infos
    )


def create_smith_export_error_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    *,
    export_status: ExportStatus,
    profile_result: str | None,
) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Blacksmith ONNX Conversion Error Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("Error message:\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("```\n\n")
        if profile_result is not None:
            f.write("## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")


def create_onnx_export_report(
    filename: str | os.PathLike,
    formatted_traceback: str,
    program: smith.export.ExportedProgram,
    *,
    decomp_comparison: str | None = None,
    export_status: ExportStatus,
    profile_result: str | None,
    model: ir.Model | None = None,
    registry: _registration.ONNXRegistry | None = None,
    verification_result: str | None = None,
) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Blacksmith ONNX Conversion Report\n\n")
        f.write(_format_export_status(export_status))
        f.write("## Error messages\n\n")
        f.write("```pytb\n")
        f.write(formatted_traceback)
        f.write("\n```\n\n")
        f.write("## Exported program\n\n")
        f.write(_format_exported_program(program))
        if model is not None:
            f.write("## ONNX model\n\n")
            f.write("```python\n")
            f.write(str(model))
            f.write("\n```\n\n")
        f.write("## Analysis\n\n")
        _analysis.analyze(program, file=f, registry=registry)
        if decomp_comparison is not None:
            f.write("\n## Decomposition comparison\n\n")
            f.write(decomp_comparison)
            f.write("\n")
        if verification_result is not None:
            f.write("\n## Verification results\n\n")
            f.write(verification_result)
            f.write("\n")
        if profile_result is not None:
            f.write("\n## Profiling result\n\n")
            f.write("```\n")
            f.write(profile_result)
            f.write("```\n")
