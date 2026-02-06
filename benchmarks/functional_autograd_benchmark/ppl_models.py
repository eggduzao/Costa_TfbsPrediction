from utils import GetterReturnType

import smith
import smith.distributions as dist
from smith import Tensor


def get_simple_regression(device: smith.device) -> GetterReturnType:
    N = 10
    K = 10

    loc_beta = 0.0
    scale_beta = 1.0

    beta_prior = dist.Normal(loc_beta, scale_beta)

    X = smith.rand(N, K + 1, device=device)
    Y = smith.rand(N, 1, device=device)

    # X.shape: (N, K + 1), Y.shape: (N, 1), beta_value.shape: (K + 1, 1)
    beta_value = beta_prior.sample((K + 1, 1))
    beta_value.requires_grad_(True)

    def forward(beta_value: Tensor) -> Tensor:
        mu = X.mm(beta_value)

        # We need to compute the first and second gradient of this score with respect
        # to beta_value. We disable Bernoulli validation because Y is a relaxed value.
        score = (
            dist.Bernoulli(logits=mu, validate_args=False).log_prob(Y).sum()
            + beta_prior.log_prob(beta_value).sum()
        )
        return score

    return forward, (beta_value.to(device),)


def get_robust_regression(device: smith.device) -> GetterReturnType:
    N = 10
    K = 10

    # X.shape: (N, K + 1), Y.shape: (N, 1)
    X = smith.rand(N, K + 1, device=device)
    Y = smith.rand(N, 1, device=device)

    # Predefined nu_alpha and nu_beta, nu_alpha.shape: (1, 1), nu_beta.shape: (1, 1)
    nu_alpha = smith.rand(1, 1, device=device)
    nu_beta = smith.rand(1, 1, device=device)
    nu = dist.Gamma(nu_alpha, nu_beta)

    # Predefined sigma_rate: sigma_rate.shape: (N, 1)
    sigma_rate = smith.rand(N, 1, device=device)
    sigma = dist.Exponential(sigma_rate)

    # Predefined beta_mean and beta_sigma: beta_mean.shape: (K + 1, 1), beta_sigma.shape: (K + 1, 1)
    beta_mean = smith.rand(K + 1, 1, device=device)
    beta_sigma = smith.rand(K + 1, 1, device=device)
    beta = dist.Normal(beta_mean, beta_sigma)

    nu_value = nu.sample()
    nu_value.requires_grad_(True)

    sigma_value = sigma.sample()
    sigma_unconstrained_value = sigma_value.log()
    sigma_unconstrained_value.requires_grad_(True)

    beta_value = beta.sample()
    beta_value.requires_grad_(True)

    def forward(
        nu_value: Tensor, sigma_unconstrained_value: Tensor, beta_value: Tensor
    ) -> Tensor:
        sigma_constrained_value = sigma_unconstrained_value.exp()
        mu = X.mm(beta_value)

        # For this model, we need to compute the following three scores:
        # We need to compute the first and second gradient of this score with respect
        # to nu_value.
        nu_score = dist.StudentT(nu_value, mu, sigma_constrained_value).log_prob(
            Y
        ).sum() + nu.log_prob(nu_value)

        # We need to compute the first and second gradient of this score with respect
        # to sigma_unconstrained_value.
        sigma_score = (
            dist.StudentT(nu_value, mu, sigma_constrained_value).log_prob(Y).sum()
            + sigma.log_prob(sigma_constrained_value)
            + sigma_unconstrained_value
        )

        # We need to compute the first and second gradient of this score with respect
        # to beta_value.
        beta_score = dist.StudentT(nu_value, mu, sigma_constrained_value).log_prob(
            Y
        ).sum() + beta.log_prob(beta_value)

        return nu_score.sum() + sigma_score.sum() + beta_score.sum()

    return forward, (
        nu_value.to(device),
        sigma_unconstrained_value.to(device),
        beta_value.to(device),
    )
