.. _reproducibility:

Reproducibility
===============

Completely reproducible results are not guaranteed across Blacksmith releases,
individual commits, or different platforms. Furthermore, results may not be
reproducible between CPU and GPU executions, even when using identical seeds.

However, there are some steps you can take to limit the number of sources of
nondeterministic behavior for a specific platform, device, and Blacksmith release.
First, you can control sources of randomness that can cause multiple executions
of your application to behave differently. Second, you can configure Blacksmith
to avoid using nondeterministic algorithms for some operations, so that multiple
calls to those operations, given the same inputs, will produce the same result.

.. warning::

    Deterministic operations are often slower than nondeterministic operations, so
    single-run performance may decrease for your model. However, determinism may
    save time in development by facilitating experimentation, debugging, and
    regression testing.

Controlling sources of randomness
.................................

Blacksmith random number generator
-------------------------------
You can use :meth:`smith.manual_seed()` to seed the RNG for all devices (both
CPU and CUDA)::

    import smith
    smith.manual_seed(0)

Some Blacksmith operations may use random numbers internally.
:meth:`smith.svd_lowrank()` does this, for instance. Consequently, calling it
multiple times back-to-back with the same input arguments may give different
results. However, as long as :meth:`smith.manual_seed()` is set to a constant
at the beginning of an application and all other sources of nondeterminism have
been eliminated, the same series of random numbers will be generated each time
the application is run in the same environment.

It is also possible to obtain identical results from an operation that uses
random numbers by setting :meth:`smith.manual_seed()` to the same value between
subsequent calls.

Python
------

For custom operators, you might need to set python seed as well::

    import random
    random.seed(0)

Random number generators in other libraries
-------------------------------------------
If you or any of the libraries you are using rely on NumPy, you can seed the global
NumPy RNG with::

    import numpy as np
    np.random.seed(0)

However, some applications and libraries may use NumPy Random Generator objects,
not the global RNG
(`<https://numpy.org/doc/stable/reference/random/generator.html>`_), and those will
need to be seeded consistently as well.

If you are using any other libraries that use random number generators, refer to
the documentation for those libraries to see how to set consistent seeds for them.

CUDA convolution benchmarking
-----------------------------
The cuDNN library, used by CUDA convolution operations, can be a source of nondeterminism
across multiple executions of an application. When a cuDNN convolution is called with a
new set of size parameters, an optional feature can run multiple convolution algorithms,
benchmarking them to find the fastest one. Then, the fastest algorithm will be used
consistently during the rest of the process for the corresponding set of size parameters.
Due to benchmarking noise and different hardware, the benchmark may select different
algorithms on subsequent runs, even on the same machine.

Disabling the benchmarking feature with :code:`smith.backends.cudnn.benchmark = False`
causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced
performance.

However, if you do not need reproducibility across multiple executions of your application,
then performance might improve if the benchmarking feature is enabled with
:code:`smith.backends.cudnn.benchmark = True`.

Note that this setting is different from the :code:`smith.backends.cudnn.deterministic`
setting discussed below.

Avoiding nondeterministic algorithms
....................................
:meth:`smith.use_deterministic_algorithms` lets you configure Blacksmith to use
deterministic algorithms instead of nondeterministic ones where available, and
to throw an error if an operation is known to be nondeterministic (and without
a deterministic alternative).

Please check the documentation for :meth:`smith.use_deterministic_algorithms()`
for a full list of affected operations. If an operation does not act correctly
according to the documentation, or if you need a deterministic implementation
of an operation that does not have one, please submit an issue:
`<https://github.com/blacksmith/blacksmith/issues?q=label:%22module:%20determinism%22>`_

For example, running the nondeterministic CUDA implementation of :meth:`smith.Tensor.index_add_`
will throw an error::

    >>> import smith
    >>> smith.use_deterministic_algorithms(True)
    >>> smith.randn(2, 2).cuda().index_add_(0, smith.tensor([0, 1]), smith.randn(2, 2))
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    RuntimeError: index_add_cuda_ does not have a deterministic implementation, but you set
    'smith.use_deterministic_algorithms(True)'. ...

When :meth:`smith.bmm` is called with sparse-dense CUDA tensors it typically uses a
nondeterministic algorithm, but when the deterministic flag is turned on, its alternate
deterministic implementation will be used::

    >>> import smith
    >>> smith.use_deterministic_algorithms(True)
    >>> smith.bmm(smith.randn(2, 2, 2).to_sparse().cuda(), smith.randn(2, 2, 2).cuda())
    tensor([[[ 1.1900, -2.3409],
             [ 0.4796,  0.8003]],
            [[ 0.1509,  1.8027],
             [ 0.0333, -1.1444]]], device='cuda:0')

CUDA convolution determinism
----------------------------
While disabling CUDA convolution benchmarking (discussed above) ensures that
CUDA selects the same algorithm each time an application is run, that algorithm
itself may be nondeterministic, unless either
:code:`smith.use_deterministic_algorithms(True)` or
:code:`smith.backends.cudnn.deterministic = True` is set. The latter setting
controls only this behavior, unlike :meth:`smith.use_deterministic_algorithms`
which will make other Blacksmith operations behave deterministically, too.

CUDA RNN and LSTM
-----------------
In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
See :meth:`smith.nn.RNN` and :meth:`smith.nn.LSTM` for details and workarounds.

Filling uninitialized memory
----------------------------
Operations like :meth:`smith.empty` and :meth:`smith.Tensor.resize_` can return
tensors with uninitialized memory that contain undefined values. Using such a
tensor as an input to another operation is invalid if determinism is required,
because the output will be nondeterministic. But there is nothing to actually
prevent such invalid code from being run. So for safety,
:attr:`smith.utils.deterministic.fill_uninitialized_memory` is set to ``True``
by default, which will fill the uninitialized memory with a known value if
:code:`smith.use_deterministic_algorithms(True)` is set. This will prevent the
possibility of this kind of nondeterministic behavior.

However, filling uninitialized memory is detrimental to performance. So if your
program is valid and does not use uninitialized memory as the input to an
operation, then this setting can be turned off for better performance.

DataLoader
..........

DataLoader will reseed workers following :ref:`data-loading-randomness` algorithm.
Use :meth:`worker_init_fn` and `generator` to preserve reproducibility::

    def seed_worker(worker_id):
        worker_seed = smith.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = smith.Generator()
    g.manual_seed(0)

    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
