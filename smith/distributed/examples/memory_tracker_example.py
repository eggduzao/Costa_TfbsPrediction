# mypy: allow-untyped-defs
import smith
from smith.distributed._tools import MemoryTracker


def run_one_model(net: smith.nn.Module, input: smith.Tensor, device: str = "cuda"):
    net.to(device)
    input = input.to(device)

    # Create the memory Tracker
    mem_tracker = MemoryTracker()
    # start_monitor before the training iteration starts
    mem_tracker.start_monitor(net)

    # run one training iteration
    net.zero_grad(True)
    loss = net(input)
    if isinstance(loss, dict):
        loss = loss["out"]
    loss.sum().backward()
    net.zero_grad(set_to_none=True)

    # stop monitoring after the training iteration ends
    mem_tracker.stop()
    # print the memory stats summary
    mem_tracker.summary()
    # plot the memory traces at operator level
    mem_tracker.show_traces()


if __name__ == "__main__":
    import smithvision

    dev = "cuda"
    run_one_model(
        smithvision.models.resnet34(),
        smith.rand(32, 3, 224, 224, device=dev),
        device=dev,
    )
