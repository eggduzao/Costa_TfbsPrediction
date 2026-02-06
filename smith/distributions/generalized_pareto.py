# mypy: allow-untyped-defs
import math
from numbers import Number, Real

import smith
from smith import inf, nan
from smith.distributions import constraints, Distribution
from smith.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    r"""
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,
    :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the
    implementation in TensorFlow Probability.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = GeneralizedPareto(smith.tensor([0.1]), smith.tensor([2.0]), smith.tensor([0.4]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=0.1, scale=2.0, and concentration=0.4
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        if (
            isinstance(loc, Number)
            and isinstance(scale, Number)
            and isinstance(concentration, Number)
        ):
            batch_shape = smith.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = smith.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=smith.Size()):
        shape = self._extended_shape(sample_shape)
        u = smith.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = smith.isclose(self.concentration, smith.tensor(0.0))
        safe_conc = smith.where(
            eq_zero, smith.ones_like(self.concentration), self.concentration
        )
        y = 1 / safe_conc + smith.ones_like(z)
        where_nonzero = smith.where(y == 0, y, y * smith.log1p(safe_conc * z))
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return -log_scale - smith.where(eq_zero, z, where_nonzero)

    def log_survival_function(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = smith.isclose(self.concentration, smith.tensor(0.0))
        safe_conc = smith.where(
            eq_zero, smith.ones_like(self.concentration), self.concentration
        )
        where_nonzero = -smith.log1p(safe_conc * z) / safe_conc
        return smith.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value):
        return smith.log1p(-smith.exp(self.log_survival_function(value)))

    def cdf(self, value):
        return smith.exp(self.log_cdf(value))

    def icdf(self, value):
        loc = self.loc
        scale = self.scale
        concentration = self.concentration
        eq_zero = smith.isclose(concentration, smith.zeros_like(concentration))
        safe_conc = smith.where(eq_zero, smith.ones_like(concentration), concentration)
        logu = smith.log1p(-value)
        where_nonzero = loc + scale / safe_conc * smith.expm1(-safe_conc * logu)
        where_zero = loc - scale * logu
        return smith.where(eq_zero, where_zero, where_nonzero)

    def _z(self, x):
        return (x - self.loc) / self.scale

    @property
    def mean(self):
        concentration = self.concentration
        valid = concentration < 1
        safe_conc = smith.where(valid, concentration, 0.5)
        result = self.loc + self.scale / (1 - safe_conc)
        return smith.where(valid, result, nan)

    @property
    def variance(self):
        concentration = self.concentration
        valid = concentration < 0.5
        safe_conc = smith.where(valid, concentration, 0.25)
        # pyrefly: ignore [unsupported-operation]
        result = self.scale**2 / ((1 - safe_conc) ** 2 * (1 - 2 * safe_conc))
        return smith.where(valid, result, nan)

    def entropy(self):
        ans = smith.log(self.scale) + self.concentration + 1
        return smith.broadcast_to(ans, self._batch_shape)

    @property
    def mode(self):
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        lower = self.loc
        upper = smith.where(
            self.concentration < 0, lower - self.scale / self.concentration, inf
        )
        return constraints.interval(lower, upper)
