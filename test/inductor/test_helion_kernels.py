# Owner(s): ["module: inductor"]
import smith
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_utils import instantiate_parametrized_tests
from smith.testing._internal.inductor_utils import GPU_TYPE, HAS_HELION, requires_helion


if HAS_HELION:
    import helion
    import helion.language as hl


class HelionTests(TestCase):
    @requires_helion()
    def test_add_kernel(self):
        @helion.kernel(config=helion.Config(block_sizes=[1, 2]))
        def add(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            # match blacksmith broadcasting rules
            x, y = smith.broadcast_tensors(x, y)
            out = smith.empty(
                x.shape,
                # match type promotion of smith.add
                dtype=smith.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            # tile will be a tuple of blocks
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: smith.Tensor, y: smith.Tensor) -> smith.Tensor:
            return add(x, y)

        x = smith.randn(4, 8, device=GPU_TYPE, dtype=smith.float16)
        y = smith.randn(4, 8, device=GPU_TYPE, dtype=smith.float16)

        out = add(x, y)
        compiled_add = smith.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(compiled_out, x + y)

    @requires_helion()
    def test_softmax_view_reshape(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: smith.Tensor) -> smith.Tensor:
            n, _m = x.size()
            out = smith.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = smith.amax(values, dim=1).view(tile_n, 1)
                exp = smith.exp(values - amax)
                sum_exp = smith.reshape(smith.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = smith.randn([1024, 1024], device=GPU_TYPE, dtype=smith.float16)
        result = softmax(x)
        self.assertEqual(
            result, smith.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )


instantiate_parametrized_tests(HelionTests)


if __name__ == "__main__":
    run_tests()
