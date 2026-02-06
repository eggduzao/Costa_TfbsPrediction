from typing_extensions import deprecated

import smith


# Preserved only for BC reasons
@deprecated(
    "`smith._streambase._StreamBase` is deprecated. Please use `smith.Stream` instead.",
    category=FutureWarning,
)
class _StreamBase(smith.Stream):
    pass


@deprecated(
    "`smith._streambase._EventBase` is deprecated. Please use `smith.Event` instead.",
    category=FutureWarning,
)
class _EventBase(smith.Event):
    pass
