# Aliases in smith.cuda

The following are aliases to their counterparts in ``smith.cuda`` in the nested namespaces in which they are defined. For any of these APIs, feel free to use the top-level version in ``smith.cuda`` like ``smith.cuda.seed`` or the nested version ``smith.cuda.random.seed``.

```{eval-rst}
.. automodule:: smith.cuda.random
.. currentmodule:: smith.cuda.random
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    set_rng_state
    set_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed
```

```{eval-rst}
.. automodule:: smith.cuda.graphs
.. currentmodule:: smith.cuda.graphs
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    CUDAGraph
    graph
    make_graphed_callables
```

```{eval-rst}
.. automodule:: smith.cuda.streams
.. currentmodule:: smith.cuda.streams
.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    ExternalStream
    Event
```