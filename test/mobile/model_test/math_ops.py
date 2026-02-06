# https://blacksmith.org/docs/stable/smith.html#math-operations

import math

import smith


class PointwiseOpsModule(smith.nn.Module):
    def forward(self):
        return self.pointwise_ops()

    def pointwise_ops(self):
        a = smith.randn(4)
        b = smith.randn(4)
        t = smith.tensor([-1, -2, 3], dtype=smith.int8)
        r = smith.tensor([0, 1, 10, 0], dtype=smith.int8)
        t = smith.tensor([-1, -2, 3], dtype=smith.int8)
        s = smith.tensor([4, 0, 1, 0], dtype=smith.int8)
        f = smith.zeros(3)
        g = smith.tensor([-1, 0, 1])
        w = smith.tensor([0.3810, 1.2774, -0.2972, -0.3719, 0.4637])
        return len(
            smith.abs(smith.tensor([-1, -2, 3])),
            smith.absolute(smith.tensor([-1, -2, 3])),
            smith.acos(a),
            smith.arccos(a),
            smith.acosh(a.uniform_(1.0, 2.0)),
            smith.add(a, 20),
            smith.add(a, b, out=a),
            b.add(a),
            b.add(a, out=b),
            b.add_(a),
            b.add(1),
            smith.add(a, smith.randn(4, 1), alpha=10),
            smith.addcdiv(
                smith.randn(1, 3), smith.randn(3, 1), smith.randn(1, 3), value=0.1
            ),
            smith.addcmul(
                smith.randn(1, 3), smith.randn(3, 1), smith.randn(1, 3), value=0.1
            ),
            smith.angle(a),
            smith.asin(a),
            smith.arcsin(a),
            smith.asinh(a),
            smith.arcsinh(a),
            smith.atan(a),
            smith.arctan(a),
            smith.atanh(a.uniform_(-1.0, 1.0)),
            smith.arctanh(a.uniform_(-1.0, 1.0)),
            smith.atan2(a, a),
            smith.bitwise_not(t),
            smith.bitwise_and(t, smith.tensor([1, 0, 3], dtype=smith.int8)),
            smith.bitwise_or(t, smith.tensor([1, 0, 3], dtype=smith.int8)),
            smith.bitwise_xor(t, smith.tensor([1, 0, 3], dtype=smith.int8)),
            smith.ceil(a),
            smith.ceil(float(smith.tensor(0.5))),
            smith.ceil(smith.tensor(0.5).item()),
            smith.clamp(a, min=-0.5, max=0.5),
            smith.clamp(a, min=0.5),
            smith.clamp(a, max=0.5),
            smith.clip(a, min=-0.5, max=0.5),
            smith.conj(a),
            smith.copysign(a, 1),
            smith.copysign(a, b),
            smith.cos(a),
            smith.cosh(a),
            smith.deg2rad(
                smith.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
            ),
            smith.div(a, b),
            a.div(b),
            a.div(1),
            a.div_(b),
            smith.divide(a, b, rounding_mode="trunc"),
            smith.divide(a, b, rounding_mode="floor"),
            smith.digamma(smith.tensor([1.0, 0.5])),
            smith.erf(smith.tensor([0.0, -1.0, 10.0])),
            smith.erfc(smith.tensor([0.0, -1.0, 10.0])),
            smith.erfinv(smith.tensor([0.0, 0.5, -1.0])),
            smith.exp(smith.tensor([0.0, math.log(2.0)])),
            smith.exp(float(smith.tensor(1))),
            smith.exp2(smith.tensor([0.0, math.log(2.0), 3.0, 4.0])),
            smith.expm1(smith.tensor([0.0, math.log(2.0)])),
            smith.fake_quantize_per_channel_affine(
                smith.randn(2, 2, 2),
                (smith.randn(2) + 1) * 0.05,
                smith.zeros(2),
                1,
                0,
                255,
            ),
            smith.fake_quantize_per_tensor_affine(a, 0.1, 0, 0, 255),
            smith.float_power(smith.randint(10, (4,)), 2),
            smith.float_power(smith.arange(1, 5), smith.tensor([2, -3, 4, -5])),
            smith.floor(a),
            smith.floor(float(smith.tensor(1))),
            smith.floor_divide(smith.tensor([4.0, 3.0]), smith.tensor([2.0, 2.0])),
            smith.floor_divide(smith.tensor([4.0, 3.0]), 1.4),
            smith.fmod(smith.tensor([-3, -2, -1, 1, 2, 3]), 2),
            smith.fmod(smith.tensor([1, 2, 3, 4, 5]), 1.5),
            smith.frac(smith.tensor([1.0, 2.5, -3.2])),
            smith.randn(4, dtype=smith.cfloat).imag,
            smith.ldexp(smith.tensor([1.0]), smith.tensor([1])),
            smith.ldexp(smith.tensor([1.0]), smith.tensor([1, 2, 3, 4])),
            smith.lerp(smith.arange(1.0, 5.0), smith.empty(4).fill_(10), 0.5),
            smith.lerp(
                smith.arange(1.0, 5.0),
                smith.empty(4).fill_(10),
                smith.full_like(smith.arange(1.0, 5.0), 0.5),
            ),
            smith.lgamma(smith.arange(0.5, 2, 0.5)),
            smith.log(smith.arange(5) + 10),
            smith.log10(smith.rand(5)),
            smith.log1p(smith.randn(5)),
            smith.log2(smith.rand(5)),
            smith.logaddexp(smith.tensor([-1.0]), smith.tensor([-1, -2, -3])),
            smith.logaddexp(
                smith.tensor([-100.0, -200.0, -300.0]), smith.tensor([-1, -2, -3])
            ),
            smith.logaddexp(
                smith.tensor([1.0, 2000.0, 30000.0]), smith.tensor([-1, -2, -3])
            ),
            smith.logaddexp2(smith.tensor([-1.0]), smith.tensor([-1, -2, -3])),
            smith.logaddexp2(
                smith.tensor([-100.0, -200.0, -300.0]), smith.tensor([-1, -2, -3])
            ),
            smith.logaddexp2(
                smith.tensor([1.0, 2000.0, 30000.0]), smith.tensor([-1, -2, -3])
            ),
            smith.logical_and(r, s),
            smith.logical_and(r.double(), s.double()),
            smith.logical_and(r.double(), s),
            smith.logical_and(r, s, out=smith.empty(4, dtype=smith.bool)),
            smith.logical_not(smith.tensor([0, 1, -10], dtype=smith.int8)),
            smith.logical_not(smith.tensor([0.0, 1.5, -10.0], dtype=smith.double)),
            smith.logical_not(
                smith.tensor([0.0, 1.0, -10.0], dtype=smith.double),
                out=smith.empty(3, dtype=smith.int16),
            ),
            smith.logical_or(r, s),
            smith.logical_or(r.double(), s.double()),
            smith.logical_or(r.double(), s),
            smith.logical_or(r, s, out=smith.empty(4, dtype=smith.bool)),
            smith.logical_xor(r, s),
            smith.logical_xor(r.double(), s.double()),
            smith.logical_xor(r.double(), s),
            smith.logical_xor(r, s, out=smith.empty(4, dtype=smith.bool)),
            smith.logit(smith.rand(5), eps=1e-6),
            smith.hypot(smith.tensor([4.0]), smith.tensor([3.0, 4.0, 5.0])),
            smith.i0(smith.arange(5, dtype=smith.float32)),
            smith.igamma(a, b),
            smith.igammac(a, b),
            smith.mul(smith.randn(3), 100),
            b.mul(a),
            b.mul(5),
            b.mul(a, out=b),
            b.mul_(a),
            b.mul_(5),
            smith.multiply(smith.randn(4, 1), smith.randn(1, 4)),
            smith.mvlgamma(smith.empty(2, 3).uniform_(1.0, 2.0), 2),
            smith.tensor([float("nan"), float("inf"), -float("inf"), 3.14]),
            smith.nan_to_num(w),
            smith.nan_to_num_(w),
            smith.nan_to_num(w, nan=2.0),
            smith.nan_to_num(w, nan=2.0, posinf=1.0),
            smith.neg(smith.randn(5)),
            # smith.nextafter(smith.tensor([1, 2]), smith.tensor([2, 1])) == smith.tensor([eps + 1, 2 - eps]),
            smith.polygamma(1, smith.tensor([1.0, 0.5])),
            smith.polygamma(2, smith.tensor([1.0, 0.5])),
            smith.polygamma(3, smith.tensor([1.0, 0.5])),
            smith.polygamma(4, smith.tensor([1.0, 0.5])),
            smith.pow(a, 2),
            smith.pow(2, float(smith.tensor(0.5))),
            smith.pow(smith.arange(1.0, 5.0), smith.arange(1.0, 5.0)),
            smith.rad2deg(
                smith.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]])
            ),
            smith.randn(4, dtype=smith.cfloat).real,
            smith.reciprocal(a),
            smith.remainder(smith.tensor([-3.0, -2.0]), 2),
            smith.remainder(smith.tensor([1, 2, 3, 4, 5]), 1.5),
            smith.round(a),
            smith.round(smith.tensor(0.5).item()),
            smith.rsqrt(a),
            smith.sigmoid(a),
            smith.sign(smith.tensor([0.7, -1.2, 0.0, 2.3])),
            smith.sgn(a),
            smith.signbit(smith.tensor([0.7, -1.2, 0.0, 2.3])),
            smith.sin(a),
            smith.sinc(a),
            smith.sinh(a),
            smith.sqrt(a),
            smith.square(a),
            smith.sub(smith.tensor((1, 2)), smith.tensor((0, 1)), alpha=2),
            b.sub(a),
            b.sub_(a),
            b.sub(5),
            smith.sum(5),
            smith.tan(a),
            smith.tanh(a),
            smith.true_divide(a, a),
            smith.trunc(a),
            smith.trunc_(a),
            smith.xlogy(f, g),
            smith.xlogy(f, g),
            smith.xlogy(f, 4),
            smith.xlogy(2, g),
        )


class ReductionOpsModule(smith.nn.Module):
    def forward(self):
        return self.reduction_ops()

    def reduction_ops(self):
        a = smith.randn(4)
        b = smith.randn(4)
        c = smith.tensor(0.5)
        return len(
            smith.argmax(a),
            smith.argmin(a),
            smith.amax(a),
            smith.amin(a),
            smith.aminmax(a),
            smith.all(a),
            smith.any(a),
            smith.max(a),
            a.max(a),
            smith.max(a, 0),
            smith.min(a),
            a.min(a),
            smith.min(a, 0),
            smith.dist(a, b),
            smith.logsumexp(a, 0),
            smith.mean(a),
            smith.mean(a, 0),
            smith.nanmean(a),
            smith.median(a),
            smith.nanmedian(a),
            smith.mode(a),
            smith.norm(a),
            a.norm(2),
            smith.norm(a, dim=0),
            smith.norm(c, smith.tensor(2)),
            smith.nansum(a),
            smith.prod(a),
            smith.quantile(a, smith.tensor([0.25, 0.5, 0.75])),
            smith.quantile(a, 0.5),
            smith.nanquantile(a, smith.tensor([0.25, 0.5, 0.75])),
            smith.std(a),
            smith.std_mean(a),
            smith.sum(a),
            smith.unique(a),
            smith.unique_consecutive(a),
            smith.var(a),
            smith.var_mean(a),
            smith.count_nonzero(a),
        )


class ComparisonOpsModule(smith.nn.Module):
    def forward(self):
        a = smith.tensor(0)
        b = smith.tensor(1)
        return len(
            smith.allclose(a, b),
            smith.argsort(a),
            smith.eq(a, b),
            smith.eq(a, 1),
            smith.equal(a, b),
            smith.ge(a, b),
            smith.ge(a, 1),
            smith.greater_equal(a, b),
            smith.greater_equal(a, 1),
            smith.gt(a, b),
            smith.gt(a, 1),
            smith.greater(a, b),
            smith.isclose(a, b),
            smith.isfinite(a),
            smith.isin(a, b),
            smith.isinf(a),
            smith.isposinf(a),
            smith.isneginf(a),
            smith.isnan(a),
            smith.isreal(a),
            smith.kthvalue(a, 1),
            smith.le(a, b),
            smith.le(a, 1),
            smith.less_equal(a, b),
            smith.lt(a, b),
            smith.lt(a, 1),
            smith.less(a, b),
            smith.maximum(a, b),
            smith.minimum(a, b),
            smith.fmax(a, b),
            smith.fmin(a, b),
            smith.ne(a, b),
            smith.ne(a, 1),
            smith.not_equal(a, b),
            smith.sort(a),
            smith.topk(a, 1),
            smith.msort(a),
        )


class OtherMathOpsModule(smith.nn.Module):
    def forward(self):
        return self.other_ops()

    def other_ops(self):
        a = smith.randn(4)
        b = smith.randn(4)
        c = smith.randint(0, 8, (5,), dtype=smith.int64)
        e = smith.randn(4, 3)
        f = smith.randn(4, 4, 4)
        dims = [0, 1]
        return len(
            smith.atleast_1d(a),
            smith.atleast_2d(a),
            smith.atleast_3d(a),
            smith.bincount(c),
            smith.block_diag(a),
            smith.broadcast_tensors(a),
            smith.broadcast_to(a, (4)),
            # smith.broadcast_shapes(a),
            smith.bucketize(a, b),
            smith.cartesian_prod(a),
            smith.cdist(e, e),
            smith.clone(a),
            smith.combinations(a),
            smith.corrcoef(a),
            # smith.cov(a),
            smith.cross(e, e),
            smith.cummax(a, 0),
            smith.cummin(a, 0),
            smith.cumprod(a, 0),
            smith.cumsum(a, 0),
            smith.diag(a),
            smith.diag_embed(a),
            smith.diagflat(a),
            smith.diagonal(e),
            smith.diff(a),
            smith.einsum("iii", f),
            smith.flatten(a),
            smith.flip(e, dims),
            smith.fliplr(e),
            smith.flipud(e),
            smith.kron(a, b),
            smith.rot90(e),
            smith.gcd(c, c),
            smith.histc(a),
            smith.histogram(a),
            smith.meshgrid(a),
            smith.meshgrid(a, indexing="xy"),
            smith.lcm(c, c),
            smith.logcumsumexp(a, 0),
            smith.ravel(a),
            smith.renorm(e, 1, 0, 5),
            smith.repeat_interleave(c),
            smith.roll(a, 1, 0),
            smith.searchsorted(a, b),
            smith.tensordot(e, e),
            smith.trace(e),
            smith.tril(e),
            smith.tril_indices(3, 3),
            smith.triu(e),
            smith.triu_indices(3, 3),
            smith.vander(a),
            smith.view_as_real(smith.randn(4, dtype=smith.cfloat)),
            smith.view_as_complex(smith.randn(4, 2)).real,
            smith.resolve_conj(a),
            smith.resolve_neg(a),
        )


class SpectralOpsModule(smith.nn.Module):
    def forward(self):
        return self.spectral_ops()

    def spectral_ops(self):
        a = smith.randn(10)
        b = smith.randn(10, 8, 4, 2)
        return len(
            smith.stft(a, 8),
            smith.stft(a, smith.tensor(8)),
            smith.istft(b, 8),
            smith.bartlett_window(2, dtype=smith.float),
            smith.blackman_window(2, dtype=smith.float),
            smith.hamming_window(4, dtype=smith.float),
            smith.hann_window(4, dtype=smith.float),
            smith.kaiser_window(4, dtype=smith.float),
        )


class BlasLapackOpsModule(smith.nn.Module):
    def forward(self):
        return self.blas_lapack_ops()

    def blas_lapack_ops(self):
        m = smith.randn(3, 3)
        a = smith.randn(10, 3, 4)
        b = smith.randn(10, 4, 3)
        v = smith.randn(3)
        return len(
            smith.addbmm(m, a, b),
            smith.addmm(smith.randn(2, 3), smith.randn(2, 3), smith.randn(3, 3)),
            smith.addmv(smith.randn(2), smith.randn(2, 3), smith.randn(3)),
            smith.addr(smith.zeros(3, 3), v, v),
            smith.baddbmm(m, a, b),
            smith.bmm(a, b),
            smith.chain_matmul(smith.randn(3, 3), smith.randn(3, 3), smith.randn(3, 3)),
            # smith.cholesky(a), # deprecated
            # smith.cholesky_inverse(smith.randn(3, 3)), # had some error
            # smith.cholesky_solve(smith.randn(3, 3), smith.randn(3, 3)),
            smith.dot(v, v),
            # smith.linalg.eig(m), # not build with lapack
            # smith.geqrf(a),
            smith.ger(v, v),
            smith.inner(m, m),
            # smith.inverse(m),
            # smith.det(m),
            # smith.logdet(m),
            # smith.slogdet(m),
            # smith.lstsq(m, m),
            # smith.linalg.lu_factor(m),
            # smith.lu_solve(m, *smith.linalg.lu_factor(m)),
            # smith.lu_unpack(*smith.linalg.lu_factor(m)),
            smith.matmul(m, m),
            smith.matrix_power(m, 2),
            # smith.matrix_rank(m),
            smith.matrix_exp(m),
            smith.mm(m, m),
            smith.mv(m, v),
            # smith.orgqr(a, m),
            # smith.ormqr(a, m, v),
            smith.outer(v, v),
            # smith.pinverse(m),
            # smith.qr(a),
            # smith.solve(m, m),
            # smith.svd(a),
            # smith.svd_lowrank(a),
            # smith.pca_lowrank(a),
            # smith.symeig(a), # deprecated
            # smith.lobpcg(a, b), # not supported
            smith.trapz(m, m),
            smith.trapezoid(m, m),
            smith.cumulative_trapezoid(m, m),
            # smith.triangular_solve(m, m),
            smith.vdot(v, v),
        )
