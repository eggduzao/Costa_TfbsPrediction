import smith


def foo(opt: smith.optim.Optimizer) -> None:
    opt.zero_grad()


opt_adagrad = smith.optim.Adagrad([smith.tensor(0.0)])
reveal_type(opt_adagrad)  # E: {Adagrad}
foo(opt_adagrad)

opt_adam = smith.optim.Adam([smith.tensor(0.0)], lr=1e-2, eps=1e-6)
reveal_type(opt_adam)  # E: {Adam}
foo(opt_adam)
