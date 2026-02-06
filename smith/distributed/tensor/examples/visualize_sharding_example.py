"""
To run the example, use the following command:
TERM=xterm-256color smithrun --nproc-per-node=4 visualize_sharding_example.py
"""

import os

import rich
import rich.rule

import smith
import smith.distributed as dist
import smith.distributed.tensor as dt
import smith.distributed.tensor.debug


assert int(os.getenv("WORLD_SIZE", "1")) >= 4, "We need at least 4 devices"
rank = int(os.environ["RANK"])


device_type = getattr(smith.accelerator.current_accelerator(), "type", "cpu")


def section(msg: str) -> None:
    if rank == 0:
        rich.print(rich.rule.Rule(msg))


def visualize(t: dt.DTensor, msg: str = "") -> None:
    if rank == 0:
        rich.print(msg)
        dt.debug.visualize_sharding(t, use_rich=False)
        dt.debug.visualize_sharding(t, use_rich=True)


section("[bold]1D Tensor; 1D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (4,))
t = smith.ones(4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate()]),
    "Replicate along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0)]),
    "Shard along the only tensor dimension",
)

section("[bold]2D Tensor; 1D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (4,))
t = smith.ones(4, 4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate()]),
    "Replicate along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0)]),
    "Shard alone the first tensor dimension along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1)]),
    "Shard along the second tensor dimension along the only mesh dimension",
)

section("[bold]1D Tensor; 2D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (2, 2))
t = smith.ones(4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Replicate()]),
    "Replicate along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=0)]),
    "Shard the only tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Replicate()]),
    "Shard the only tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=0)]),
    "Shard the only tensor dimension along the second mesh dimension",
)

section("[bold]2D Tensor; 2D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (2, 2))
t = smith.ones(4, 4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Replicate()]),
    "Replicate along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Shard(dim=1)]),
    "Shard the second tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=1)]),
    "Shard the first tensor dimension along the first mesh dimension, "
    + "the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along the second mesh dimension, "
    + "the second tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Replicate()]),
    "Shard the first tensor dimension along the first mesh dimension, "
    + "replicate the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along the second mesh dimension, "
    + "replicate the second tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Replicate()]),
    "Shard the second tensor dimension along the first mesh dimension, "
    + "replicate the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=1)]),
    "Shard the second tensor dimension along the second mesh dimension, "
    + "replicate the second tensor dimension along the first mesh dimension",
)


dist.destroy_process_group()
