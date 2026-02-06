# mypy: allow-untyped-defs
import math

import smith
from smith import Tensor
from smith.distributions import constraints
from smith.distributions.exp_family import ExponentialFamily
from smith.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from smith.nn.functional import binary_cross_entropy_with_logits
from smith.types import _Number, _size, Number


__all__ = ["ContinuousBernoulli"]


class ContinuousBernoulli(ExponentialFamily):
    r"""
    Creates a continuous Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    The distribution is supported in [0, 1] and parameterized by 'probs' (in
    (0,1)) or 'logits' (real-valued). Note that, unlike the Bernoulli, 'probs'
    does not correspond to a probability and 'logits' does not correspond to
    log-odds, but the same names are used due to the similarity with the
    Bernoulli. See [1] for more details.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = ContinuousBernoulli(smith.tensor([0.3]))
        >>> m.sample()
        tensor([ 0.2538])

    Args:
        probs (Number, Tensor): (0,1) valued parameters
        logits (Number, Tensor): real valued parameters whose sigmoid matches 'probs'

    [1] The continuous Bernoulli: fixing a pervasive error in variational
    autoencoders, Loaiza-Ganem G and Cunningham JP, NeurIPS 2019.
    https://arxiv.org/abs/1907.06845
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    _mean_carrier_measure = 0
    has_rsample = True

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        lims: tuple[float, float] = (0.499, 0.501),
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
            # validate 'probs' here if necessary as it is later clamped for numerical stability
            # close to 0 and 1, later on; otherwise the clamped 'probs' would always pass
            if validate_args is not None:
                if not self.arg_constraints["probs"].check(self.probs).all():
                    raise ValueError("The parameter probs has invalid values")
            # pyrefly: ignore [read-only]
            self.probs = clamp_probs(self.probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            is_scalar = isinstance(logits, _Number)
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = smith.Size()
        else:
            batch_shape = self._param.size()
        self._lims = lims
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ContinuousBernoulli, _instance)
        new._lims = self._lims
        batch_shape = smith.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(ContinuousBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    def _outside_unstable_region(self):
        return smith.max(
            smith.le(self.probs, self._lims[0]), smith.gt(self.probs, self._lims[1])
        )

    def _cut_probs(self):
        return smith.where(
            self._outside_unstable_region(),
            self.probs,
            self._lims[0] * smith.ones_like(self.probs),
        )

    def _cont_bern_log_norm(self):
        """computes the log normalizing constant as a function of the 'probs' parameter"""
        cut_probs = self._cut_probs()
        cut_probs_below_half = smith.where(
            smith.le(cut_probs, 0.5), cut_probs, smith.zeros_like(cut_probs)
        )
        cut_probs_above_half = smith.where(
            smith.ge(cut_probs, 0.5), cut_probs, smith.ones_like(cut_probs)
        )
        log_norm = smith.log(
            smith.abs(smith.log1p(-cut_probs) - smith.log(cut_probs))
        ) - smith.where(
            smith.le(cut_probs, 0.5),
            smith.log1p(-2.0 * cut_probs_below_half),
            smith.log(2.0 * cut_probs_above_half - 1.0),
        )
        x = smith.pow(self.probs - 0.5, 2)
        taylor = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x
        return smith.where(self._outside_unstable_region(), log_norm, taylor)

    @property
    def mean(self) -> Tensor:
        cut_probs = self._cut_probs()
        mus = cut_probs / (2.0 * cut_probs - 1.0) + 1.0 / (
            smith.log1p(-cut_probs) - smith.log(cut_probs)
        )
        x = self.probs - 0.5
        taylor = 0.5 + (1.0 / 3.0 + 16.0 / 45.0 * smith.pow(x, 2)) * x
        return smith.where(self._outside_unstable_region(), mus, taylor)

    @property
    def stddev(self) -> Tensor:
        return smith.sqrt(self.variance)

    @property
    def variance(self) -> Tensor:
        cut_probs = self._cut_probs()
        vars = cut_probs * (cut_probs - 1.0) / smith.pow(
            1.0 - 2.0 * cut_probs, 2
        ) + 1.0 / smith.pow(smith.log1p(-cut_probs) - smith.log(cut_probs), 2)
        x = smith.pow(self.probs - 0.5, 2)
        taylor = 1.0 / 12.0 - (1.0 / 15.0 - 128.0 / 945.0 * x) * x
        return smith.where(self._outside_unstable_region(), vars, taylor)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return clamp_probs(logits_to_probs(self.logits, is_binary=True))

    @property
    def param_shape(self) -> smith.Size:
        return self._param.size()

    def sample(self, sample_shape=smith.Size()):
        shape = self._extended_shape(sample_shape)
        u = smith.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        with smith.no_grad():
            return self.icdf(u)

    def rsample(self, sample_shape: _size = smith.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = smith.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return (
            -binary_cross_entropy_with_logits(logits, value, reduction="none")
            + self._cont_bern_log_norm()
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cut_probs = self._cut_probs()
        cdfs = (
            smith.pow(cut_probs, value) * smith.pow(1.0 - cut_probs, 1.0 - value)
            + cut_probs
            - 1.0
        ) / (2.0 * cut_probs - 1.0)
        unbounded_cdfs = smith.where(self._outside_unstable_region(), cdfs, value)
        return smith.where(
            smith.le(value, 0.0),
            smith.zeros_like(value),
            smith.where(smith.ge(value, 1.0), smith.ones_like(value), unbounded_cdfs),
        )

    def icdf(self, value):
        cut_probs = self._cut_probs()
        return smith.where(
            self._outside_unstable_region(),
            (
                smith.log1p(-cut_probs + value * (2.0 * cut_probs - 1.0))
                - smith.log1p(-cut_probs)
            )
            / (smith.log(cut_probs) - smith.log1p(-cut_probs)),
            value,
        )

    def entropy(self):
        log_probs0 = smith.log1p(-self.probs)
        log_probs1 = smith.log(self.probs)
        return (
            self.mean * (log_probs0 - log_probs1)
            - self._cont_bern_log_norm()
            - log_probs0
        )

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.logits,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        """computes the log normalizing constant as a function of the natural parameter"""
        out_unst_reg = smith.max(
            smith.le(x, self._lims[0] - 0.5), smith.gt(x, self._lims[1] - 0.5)
        )
        cut_nat_params = smith.where(
            out_unst_reg, x, (self._lims[0] - 0.5) * smith.ones_like(x)
        )
        log_norm = smith.log(
            smith.abs(smith.special.expm1(cut_nat_params))
        ) - smith.log(smith.abs(cut_nat_params))
        taylor = 0.5 * x + smith.pow(x, 2) / 24.0 - smith.pow(x, 4) / 2880.0
        return smith.where(out_unst_reg, log_norm, taylor)
