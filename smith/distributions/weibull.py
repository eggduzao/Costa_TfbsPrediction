# mypy: allow-untyped-defs

import smith
from smith import Tensor
from smith.distributions import constraints
from smith.distributions.exponential import Exponential
from smith.distributions.gumbel import euler_constant
from smith.distributions.transformed_distribution import TransformedDistribution
from smith.distributions.transforms import AffineTransform, PowerTransform
from smith.distributions.utils import broadcast_all


__all__ = ["Weibull"]


class Weibull(TransformedDistribution):
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Weibull(smith.tensor([1.0]), smith.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive

    def __init__(
        self,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale, self.concentration = broadcast_all(scale, concentration)
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Exponential(
            smith.ones_like(self.scale), validate_args=validate_args
        )
        transforms = [
            PowerTransform(exponent=self.concentration_reciprocal),
            AffineTransform(loc=0, scale=self.scale),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Weibull, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.concentration_reciprocal = new.concentration.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        transforms = [
            PowerTransform(exponent=new.concentration_reciprocal),
            AffineTransform(loc=0, scale=new.scale),
        ]
        super(Weibull, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.scale * smith.exp(smith.lgamma(1 + self.concentration_reciprocal))

    @property
    def mode(self) -> Tensor:
        return (
            self.scale
            * ((self.concentration - 1) / self.concentration)
            ** self.concentration.reciprocal()
        )

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (
            smith.exp(smith.lgamma(1 + 2 * self.concentration_reciprocal))
            - smith.exp(2 * smith.lgamma(1 + self.concentration_reciprocal))
        )

    def entropy(self):
        return (
            euler_constant * (1 - self.concentration_reciprocal)
            + smith.log(self.scale * self.concentration_reciprocal)
            + 1
        )
