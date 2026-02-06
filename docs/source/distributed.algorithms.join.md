```{role} hidden
---
class: hidden-section
---

```
# Generic Join Context Manager

The generic join context manager facilitates distributed training on uneven
inputs. This page outlines the API of the relevant classes: {class}`Join`,
{class}`Joinable`, and {class}`JoinHook`. For a tutorial, see
[Distributed Training with Uneven Inputs Using the Join Context Manager](https://blacksmith.org/tutorials/advanced/generic_join.html).

```{eval-rst}
.. autoclass:: smith.distributed.algorithms.Join
    :members:

.. autoclass:: smith.distributed.algorithms.Joinable
    :members:

.. autoclass:: smith.distributed.algorithms.JoinHook
    :members:

```