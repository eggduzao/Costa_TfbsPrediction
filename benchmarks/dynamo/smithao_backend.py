from collections.abc import Callable
from typing import Any

import smith


def setup_baseline():
    from smithao.quantization.utils import recommended_inductor_config_setter

    recommended_inductor_config_setter()
    smith._dynamo.config.automatic_dynamic_shapes = False
    smith._dynamo.config.recompile_limit = 10000


def smithao_optimize_ctx(quantization: str):
    from smithao.quantization.quant_api import (
        autoquant,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from smithao.utils import unwrap_tensor_subclass

    def inner(model_iter_fn: Callable):
        def _smithao_apply(module: smith.nn.Module, example_inputs: Any):
            if getattr(module, "_quantized", None) is None:
                if quantization == "int8dynamic":
                    quantize_(
                        module,
                        int8_dynamic_activation_int8_weight(),
                        set_inductor_config=False,
                    )
                elif quantization == "int8weightonly":
                    quantize_(module, int8_weight_only(), set_inductor_config=False)
                elif quantization == "int4weightonly":
                    quantize_(module, int4_weight_only(), set_inductor_config=False)
                if quantization == "autoquant":
                    autoquant(module, error_on_unseen=False, set_inductor_config=False)
                    if isinstance(example_inputs, dict):
                        module(**example_inputs)
                    else:
                        module(*example_inputs)
                    from smithao.quantization.autoquant import AUTOQUANT_CACHE

                    if len(AUTOQUANT_CACHE) == 0:
                        raise Exception(  # noqa: TRY002
                            "NotAutoquantizable"
                            f"Found no autoquantizable layers in model {type(module)}, stopping autoquantized run"
                        )
                else:
                    unwrap_tensor_subclass(module)
                setattr(module, "_quantized", True)  # noqa: B010
            model_iter_fn(module, example_inputs)

        return _smithao_apply

    return inner
