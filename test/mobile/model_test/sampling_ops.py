import smith


# https://blacksmith.org/docs/stable/smith.html#random-sampling


class SamplingOpsModule(smith.nn.Module):
    def forward(self):
        a = smith.empty(3, 3).uniform_(0.0, 1.0)
        size = (1, 4)
        weights = smith.tensor([0, 10, 3, 0], dtype=smith.float)
        return len(
            # smith.seed(),
            # smith.manual_seed(0),
            smith.bernoulli(a),
            # smith.initial_seed(),
            smith.multinomial(weights, 2),
            smith.normal(2.0, 3.0, size),
            smith.poisson(a),
            smith.rand(2, 3),
            smith.rand_like(a),
            smith.randint(10, size),
            smith.randint_like(a, 4),
            smith.rand(4),
            smith.randn_like(a),
            smith.randperm(4),
            a.bernoulli_(),
            a.cauchy_(),
            a.exponential_(),
            a.geometric_(0.5),
            a.log_normal_(),
            a.normal_(),
            a.random_(),
            a.uniform_(),
        )
