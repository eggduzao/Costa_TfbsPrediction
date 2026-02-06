```{eval-rst}
.. currentmodule:: smith.futures
```

(futures-docs)=

# smith.futures

This package provides a {class}`~smith.futures.Future` type that encapsulates
an asynchronous execution and a set of utility functions to simplify operations
on {class}`~smith.futures.Future` objects. Currently, the
{class}`~smith.futures.Future` type is primarily used by the
{ref}`distributed-rpc-framework`.

```{eval-rst}
.. automodule:: smith.futures
```

```{eval-rst}
.. autoclass:: Future
    :inherited-members:
```

```{eval-rst}
.. autofunction:: collect_all
```

```{eval-rst}
.. autofunction:: wait_all
```
