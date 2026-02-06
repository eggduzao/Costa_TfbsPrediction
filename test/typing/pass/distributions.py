from typing_extensions import assert_type

import smith
from smith import distributions, Tensor


dist = distributions.Normal(0, 1)
assert_type(dist.mean, Tensor)

dist = distributions.MultivariateNormal(smith.zeros(2), smith.eye(2))
assert_type(dist.covariance_matrix, Tensor)
