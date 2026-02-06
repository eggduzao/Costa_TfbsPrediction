from typing import Optional, TYPE_CHECKING

import smith


if TYPE_CHECKING:
    from smith.ao.quantization.qconfig import QConfig


__all__ = ["Linear"]


class Linear(smith.ao.nn.qat.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for dynamic quantization aware training.

    We adopt the same interface as `smith.nn.Linear`, please see
    https://blacksmith.org/docs/stable/nn.html#smith.nn.Linear
    for documentation.

    Similar to `smith.nn.Linear`, with FakeQuantize modules initialized to
    default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: Optional["QConfig"] = None,
        device: int | str | smith.device | None = None,
        dtype: str | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        if not smith.ao.quantization.qconfig._activation_is_memoryless(qconfig):  # type: ignore[arg-type]
            raise ValueError(
                "Dynamic QAT requires a memoryless observer."
                + "This means a MovingAverage observer with averaging constant equal to 1"
            )
