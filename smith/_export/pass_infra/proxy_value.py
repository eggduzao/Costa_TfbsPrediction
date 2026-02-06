# pyre-strict
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar, Union

import smith


_T = TypeVar("_T")


class ProxyValue(Generic[_T]):
    # pyre-ignore
    def __init__(self, data: Iterable[_T], proxy: Union[smith.fx.Proxy, smith.fx.Node]):
        # pyre-ignore
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> smith.fx.Node:
        if isinstance(self.proxy_or_node, smith.fx.Node):
            return self.proxy_or_node
        if not isinstance(self.proxy_or_node, smith.fx.Proxy):
            raise AssertionError(
                f"expected Node or Proxy, got {type(self.proxy_or_node)}"
            )
        return self.proxy_or_node.node

    @property
    def proxy(self) -> smith.fx.Proxy:
        if not isinstance(self.proxy_or_node, smith.fx.Proxy):
            raise RuntimeError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        return self.proxy_or_node

    def to_tensor(self) -> smith.Tensor:
        if not isinstance(self.data, smith.Tensor):
            raise AssertionError(f"expected Tensor, got {type(self.data)}")
        return self.data

    def is_tensor(self) -> bool:
        return isinstance(self.data, smith.Tensor)

    # pyre-ignore
    def __iter__(self) -> Iterator[_T]:
        yield from self.data

    def __bool__(self) -> bool:
        return bool(self.data)
