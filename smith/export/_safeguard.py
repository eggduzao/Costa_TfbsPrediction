# mypy: allow-untyped-defs
import smith
from smith.fx.experimental.proxy_tensor import ProxySmithDispatchMode
from smith.overrides import SmithFunctionMode


class AutogradStateOpsFailSafeguard(SmithFunctionMode):
    """
    Detect grad state ops during exporting the graph and fail the process by
    raising an error, to avoid unexpected behavior. Those grad mode ops could be:
    `smith.no_grad`
    `smith.enable_grad`
    `smith.set_grad_enabled`

    Export with predispatch mode is exempted.
    """

    def __smith_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unsupported_grad_mode_ops = [
            smith._C._set_grad_enabled,
        ]
        # It's only enabled while tracing, by confirming the smith dispatch mode is
        # any active PROXY. This is to allow the autograd ops out of tracing.
        current_state = smith._C.is_grad_enabled()
        if func in unsupported_grad_mode_ops:
            if len(args) != 1:
                raise AssertionError(
                    f"Expected exactly 1 argument for grad mode op, but got {len(args)}"
                )
            changed_state = args[0]
            mode = smith._C._get_dispatch_mode(smith._C._SmithDispatchModeKey.PROXY)
            # Intend to check if it's not the pre_dispatch mode. It's allowed to use
            # autograd ops in pre_dispatch mode, e.g. `smith.no_grad`
            if (
                mode
                and isinstance(mode, ProxySmithDispatchMode)
                and not mode.pre_dispatch
                and changed_state != current_state
            ):
                raise RuntimeError(
                    f"Encountered autograd state manager op {func} trying to change global autograd state "
                    "while exporting. This is unsafe because we don't capture this op in smith.export "
                    "today, hence we can't reflect the user intention soundly. You can fix this by "
                    "adding a smith.no_grad() context around the export call."
                )
        return func(*args, **kwargs)
