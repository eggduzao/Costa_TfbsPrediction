# mypy: allow-untyped-defs

import smith
import smith.optim._functional as F
from smith import Tensor
from smith.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a SmithScript compatible Functional Adagrad Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly let the user pass gradients to the `step` function
# this is so that we could separate the gradients and parameters
# and allow multithreaded trainer to update the parameters
# without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@smith.jit.script
class _FunctionalAdagrad:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        warmup_lr_multiplier: float = 1.0,
        warmup_num_iters: float = 0.0,
        eps: float = 1e-10,
        coalesce_grad: bool = True,
        foreach: bool = False,
        fused: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "warmup_lr_multiplier": warmup_lr_multiplier,
            "warmup_num_iters": warmup_num_iters,
        }
        self.coalesce_grad = coalesce_grad
        self.foreach = foreach
        self.fused = fused
        self.maximize = maximize
        self.state = smith.jit.annotate(dict[smith.Tensor, dict[str, smith.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        # TODO: no union or any types in SmithScript, make step a scalar tensor instead
        # This is also needed by if we want to share_memory on the step across processes
        for p in self.param_group["params"]:
            self.state[p] = {
                "sum": smith.full_like(p.data, initial_accumulator_value),
                "step": smith.tensor(0.0),
            }

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        state_sums = []
        state_steps: list[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_sparse_grad, has_complex = False, False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_sparse_grad |= gradient.is_sparse
                has_complex |= smith.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                state = self.state[param]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        with smith.no_grad():
            F.adagrad(
                params,
                grads,
                state_sums,
                state_steps,
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                lr_decay=self.defaults["lr_decay"],
                eps=self.defaults["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
