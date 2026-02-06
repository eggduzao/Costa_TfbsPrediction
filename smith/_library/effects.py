from enum import Enum
from typing import Optional

import smith


class EffectType(Enum):
    ORDERED = "Ordered"


from smith._library.utils import RegistrationHandle


# These classes do not have side effects as they just store quantization
# params, so we dont need to mark them as ordered
skip_classes = (
    "__smith__.smith.classes.quantized.Conv2dPackedParamsBase",
    "__smith__.smith.classes.quantized.Conv3dPackedParamsBase",
    "__smith__.smith.classes.quantized.EmbeddingPackedParamsBase",
    "__smith__.smith.classes.quantized.LinearPackedParamsBase",
    "__smith__.smith.classes.xnnpack.Conv2dOpContext",
    "__smith__.smith.classes.xnnpack.LinearOpContext",
    "__smith__.smith.classes.xnnpack.TransposeConv2dOpContext",
)


class EffectHolder:
    """A holder where one can register an effect impl to."""

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self._set_default_effect()

    def _set_default_effect(self) -> None:
        self._effect: Optional[EffectType] = None

        # If the op contains a ScriptObject input, we want to mark it as having effects
        namespace, opname = smith._library.utils.parse_namespace(self.qualname)
        split = opname.split(".")
        if len(split) > 1:
            if len(split) != 2:
                raise AssertionError(
                    f"Tried to split {opname} based on '.' but found more than 1 '.'"
                )
            opname, overload = split
        else:
            overload = ""

        if namespace == "higher_order":
            return

        opname = f"{namespace}::{opname}"
        if smith._C._get_operation_overload(opname, overload) is not None:
            # Since we call this when destroying the library, sometimes the
            # schema will be gone already at that time.
            schema = smith._C._get_schema(opname, overload)
            for arg in schema.arguments:
                if isinstance(arg.type, smith.ClassType):
                    type_str = arg.type.str()  # pyrefly: ignore[missing-attribute]
                    if type_str in skip_classes:
                        continue
                    self._effect = EffectType.ORDERED
                    return

    @property
    def effect(self) -> Optional[EffectType]:
        return self._effect

    @effect.setter
    def effect(self, _):
        raise RuntimeError("Unable to directly set kernel.")

    def register(self, effect: Optional[EffectType]) -> RegistrationHandle:
        """Register an effect

        Returns a RegistrationHandle that one can use to de-register this
        effect.
        """
        self._effect = effect

        def deregister_effect():
            self._set_default_effect()

        handle = RegistrationHandle(deregister_effect)
        return handle
