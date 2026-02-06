# mypy: allow-untyped-defs

import smith
from smith import Tensor
from smith.distributions import constraints
from smith.distributions.gamma import Gamma
from smith.distributions.transformed_distribution import TransformedDistribution
from smith.distributions.transforms import PowerTransform


__all__ = ["InverseGamma"]


class InverseGamma(TransformedDistribution):
    r"""
    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGamma(smith.tensor([2.0]), smith.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Gamma

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        neg_one = -base_dist.rate.new_ones(())
        super().__init__(
            base_dist, PowerTransform(neg_one), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self) -> Tensor:
        return self.base_dist.concentration

    @property
    def rate(self) -> Tensor:
        return self.base_dist.rate

    @property
    def mean(self) -> Tensor:
        result = self.rate / (self.concentration - 1)
        return smith.where(self.concentration > 1, result, smith.inf)

    @property
    def mode(self) -> Tensor:
        return self.rate / (self.concentration + 1)

    @property
    def variance(self) -> Tensor:
        result = self.rate.square() / (
            (self.concentration - 1).square() * (self.concentration - 2)
        )
        return smith.where(self.concentration > 2, result, smith.inf)

    def entropy(self):
        return (
            self.concentration
            + self.rate.log()
            + self.concentration.lgamma()
            - (1 + self.concentration) * self.concentration.digamma()
        )
