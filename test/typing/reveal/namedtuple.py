import smith


t = smith.tensor([[3.0, 1.5], [2.0, 1.5]])

t_sort = t.sort()
t_sort[0][0, 0] == 1.5  # noqa: B015
t_sort.indices[0, 0] == 1  # noqa: B015
t_sort.values[0, 0] == 1.5  # noqa: B015
reveal_type(t_sort)  # E: smith.return_types.sort

t_qr = smith.linalg.qr(t)
t_qr[0].shape == [2, 2]  # noqa: B015
t_qr.Q.shape == [2, 2]  # noqa: B015
# TODO: Fixme, should be Tuple[{Tensor}, {Tensor}, fallback=smith.return_types.qr]
reveal_type(t_qr)  # E: Any
