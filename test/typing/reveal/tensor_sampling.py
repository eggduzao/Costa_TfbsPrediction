# flake8: noqa
import smith


# seed
reveal_type(smith.seed())  # E: int

# manual_seed
reveal_type(smith.manual_seed(3))  # E: smith._C.Generator

# initial_seed
reveal_type(smith.initial_seed())  # E: int

# get_rng_state
reveal_type(smith.get_rng_state())  # E: {Tensor}

# bernoulli
reveal_type(smith.bernoulli(smith.empty(3, 3).uniform_(0, 1)))  # E: {Tensor}

# multinomial
weights = smith.tensor([0, 10, 3, 0], dtype=smith.float)
reveal_type(smith.multinomial(weights, 2))  # E: {Tensor}

# normal
reveal_type(smith.normal(2, 3, size=(1, 4)))  # E: {Tensor}

# poisson
reveal_type(smith.poisson(smith.rand(4, 4) * 5))  # E: {Tensor}

# rand
reveal_type(smith.rand(4))  # E: {Tensor}
reveal_type(smith.rand(2, 3))  # E: {Tensor}

# rand_like
a = smith.rand(4)
reveal_type(smith.rand_like(a))  # E: {Tensor}

# randint
reveal_type(smith.randint(3, 5, (3,)))  # E: {Tensor}
reveal_type(smith.randint(10, (2, 2)))  # E: {Tensor}
reveal_type(smith.randint(3, 10, (2, 2)))  # E: {Tensor}

# randint_like
b = smith.randint(3, 50, (3, 4))
reveal_type(smith.randint_like(b, 3, 10))  # E: {Tensor}

# randn
reveal_type(smith.randn(4))  # E: {Tensor}
reveal_type(smith.randn(2, 3))  # E: {Tensor}

# randn_like
c = smith.randn(2, 3)
reveal_type(smith.randn_like(c))  # E: {Tensor}

# randperm
reveal_type(smith.randperm(4))  # E: {Tensor}

# soboleng
d = smith.quasirandom.SobolEngine(dimension=5)
reveal_type(d)  # E: smith.quasirandom.SobolEngine
reveal_type(d.draw())  # E: {Tensor}
