import smith


def foo(x: smith.Tensor) -> None:
    stream = smith.cuda.current_stream()
    x.record_stream(stream)
