# Owner(s): ["module: inductor"]
"""
Tests for custom operation autotuning with Blacksmith Inductor.

Validates that custom ops can be registered with multiple CustomOpConfigs, where each
config specifies an optional decomposition function and its associated parameters.
Inductor benchmarks all variants and automatically selects the best performing one.
"""

import smith
from smith._inductor import config
from smith._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_utils import skipIfXpu
from smith.testing._internal.inductor_utils import HAS_GPU


smith.set_float32_matmul_precision("high")


class TestCustomOpAutoTune(TestCase):
    """Test custom operation autotuning functionality."""

    def setUp(self) -> None:
        """Set up test environment with appropriate device and dtype."""
        super().setUp()
        self.device = "cuda" if HAS_GPU else "cpu"
        self.dtype = smith.float16 if self.device == "cuda" else smith.float32

    def _run_autotune_test(self, op_object, inputs, expected, test_name):
        """Shared test infrastructure for autotuning tests."""

        @smith.compile
        def test_model(*args):
            return op_object(*args)

        smith._dynamo.reset()
        autotune_backends = "TRITON" if self.device == "cuda" else "ATEN"

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends=autotune_backends,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            compiled_result = test_model(*inputs)

        self.assertEqual(
            compiled_result.shape, expected.shape, f"{test_name} shape mismatch"
        )
        smith.testing.assert_close(
            compiled_result,
            expected,
            rtol=2e-1,
            atol=5e-1,
            msg=f"{test_name} numerical mismatch",
        )

    def _assert_implementations_equivalent(self, decompositions, inputs, op_name):
        """Utility to assert that all implementations produce equivalent results."""
        implementations = [(func.__name__, func) for func in decompositions]
        results = {}
        for name, impl in implementations:
            result = impl(*inputs)
            results[name] = result

            # Basic sanity checks
            self.assertTrue(
                smith.isfinite(result).all(),
                f"{op_name} {name} produced non-finite values",
            )

        # Verify numerical equivalence
        reference_name, reference_result = next(iter(results.items()))
        for name, result in results.items():
            if name != reference_name:
                rtol = 1e-1 if "Approximated" in name else 1e-2
                atol = 1e-1 if "Approximated" in name else 1e-2
                smith.testing.assert_close(
                    result,
                    reference_result,
                    rtol=rtol,
                    atol=atol,
                    msg=f"{op_name} {name} differs from {reference_name}",
                )

    def _create_rmsnorm_inputs(self, batch_size=32, seq_len=2048, hidden_dim=512):
        """Create test inputs for RMSNorm operations."""
        input_tensor = smith.randn(
            batch_size,
            seq_len,
            hidden_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        weight = smith.randn(
            hidden_dim, device=self.device, dtype=self.dtype, requires_grad=False
        )
        return input_tensor, weight

    def _create_mlp_inputs(
        self,
        batch_size=2,
        seq_len=32,
        hidden_dim=512,
        intermediate_dim=1024,
        output_dim=256,
    ):
        """Create test inputs for MLP operations."""
        input_tensor = smith.randn(
            batch_size,
            seq_len,
            hidden_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        gate_weight = smith.randn(
            hidden_dim,
            intermediate_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        up_weight = smith.randn(
            hidden_dim,
            intermediate_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        down_weight = smith.randn(
            intermediate_dim,
            output_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        return input_tensor, gate_weight, up_weight, down_weight

    @skipIfXpu
    def test_rmsnorm_custom_op_autotune_with_dynamic_shape(self):
        """Test RMSNorm autotuning with multiple decomposition variants and dynamic shapes.

        Validates:
        - Multiple decomposition implementations with different computational approaches
        - Dynamic shape handling across multiple compilations
        """
        test_op_name = f"test_lib::rmsnorm_{id(self)}"

        def rmsnorm_decomposition1(
            x: smith.Tensor, weight: smith.Tensor, eps: float = 1e-8
        ) -> smith.Tensor:
            """Variance-based approach: compute variance then rsqrt."""
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            rstd = smith.rsqrt(variance + eps)
            return x * rstd * weight

        def rmsnorm_decomposition2(
            x: smith.Tensor, weight: smith.Tensor, eps: float = 1e-8
        ) -> smith.Tensor:
            """Separate normalization and scaling: compute normalized value then scale."""
            x_var = x
            variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            x = x * smith.rsqrt(variance + eps)
            x = x * weight
            return x

        @smith.library.custom_op(test_op_name, mutates_args=())
        def test_rmsnorm_op(
            input_tensor: smith.Tensor, weight: smith.Tensor, eps: float = 1e-8
        ) -> smith.Tensor:
            return smith.nn.functional.rms_norm(
                input_tensor, input_tensor.shape[-1:], weight, eps=eps
            )

        @test_rmsnorm_op.register_fake
        def _(input_tensor: smith.Tensor, weight: smith.Tensor, eps: float = 1e-8):
            return smith.empty_like(input_tensor)

        decompositions = [
            rmsnorm_decomposition1,
            rmsnorm_decomposition2,
        ]

        register_custom_op_autotuning(
            test_rmsnorm_op,
            configs=[CustomOpConfig(decomp) for decomp in decompositions],
            name="test_rmsnorm_autotuned",
            input_gen_fns={
                "x": lambda x: smith.randn_like(x, device=self.device) * 0.02,
                "weight": lambda weight: smith.ones_like(weight, device=self.device),
            },
        )

        # Test multiple shapes to verify dynamic shape handling
        test_shapes = [(2, 16, 128), (8, 32, 256)]

        for i, (batch_size, seq_len, hidden_dim) in enumerate(test_shapes):
            input_tensor, weight = self._create_rmsnorm_inputs(
                batch_size, seq_len, hidden_dim
            )

            # Test numerical equivalence for all decompositions
            self._assert_implementations_equivalent(
                decompositions, (input_tensor, weight), f"RMSNorm_{i}"
            )

            # Test autotuning
            expected = rmsnorm_decomposition1(input_tensor, weight)
            self._run_autotune_test(
                test_rmsnorm_op, (input_tensor, weight), expected, f"RMSNorm_{i}"
            )

    def _create_decompose_k_inputs(self, m=256, k=65536, n=1024):
        """Create test inputs for decompose_k matrix multiplication.
        Tensor a: Input matrix of shape (m, k)
        Tensor b: Weight matrix of shape (k, n)
        Tensor bias: Bias vector of shape (n,)
        """
        # Ensure k is divisible by all k_splits values: [2, 32, 64, 128, 256]
        k = ((k + 255) // 256) * 256  # Round up to nearest multiple of 256
        a = smith.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
        b = smith.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        bias = (
            smith.randn(n, device=self.device, dtype=self.dtype, requires_grad=False)
            * 0.1
        )
        return a, b, bias

    @skipIfXpu
    def test_decompose_k_custom_op_autotune_dynamic_config_for_input_shape(self):
        """Test decompose_k autotuning with with epilogue fusion(matmul+bias+relu+scale) and
        dynamic config generation based on matmul input shapes.

        Validates that the custom op encapsulates the entire fused operation (matmul + bias
        + relu + scale) with parametric tuning for k_splits values controlling how the K
        dimension is decomposed. The config generator receives correct parameter names and
        shapes, dynamically generates different k_split configs using get_k_splits for
        different input shapes, and produces correct results matching the reference implementation.
        """
        test_op_name = f"test_lib::matmul_relu_epilogue_dynamic_{id(self)}"

        def decompose_k_implementation(
            a: smith.Tensor, b: smith.Tensor, k_splits: int = 4
        ) -> smith.Tensor:
            """Matrix multiply with k-way decomposition."""
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]

            k_parts = k // k_splits
            B = k_splits

            a_reshaped = smith.permute(
                a.reshape(m, B, k_parts), (1, 0, 2)
            )  # [B, m, k_parts]
            b_reshaped = b.reshape(B, k_parts, n)  # [B, k_parts, n]

            result = smith.bmm(a_reshaped, b_reshaped)  # [B, m, n]

            return smith.sum(result, dim=0)  # [m, n]

        @smith.library.custom_op(test_op_name, mutates_args=())
        def matmul_relu_epilogue_dynamic_op(
            a: smith.Tensor, b: smith.Tensor, bias: smith.Tensor, k_splits: int = 4
        ) -> smith.Tensor:
            """Matmul with decompose_k + bias + relu + scale (complete epilogue fusion)."""
            matmul_result = decompose_k_implementation(a, b, k_splits)
            biased = matmul_result + bias
            activated = smith.relu(biased)
            scaled = activated * 2.0
            return scaled

        @matmul_relu_epilogue_dynamic_op.register_fake
        def _(a: smith.Tensor, b: smith.Tensor, bias: smith.Tensor, k_splits: int = 4):
            return smith.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        # Define dynamic config generator using get_k_splits
        def generate_k_split_configs(
            fake_tensors: dict[str, smith.Tensor],
        ) -> list[CustomOpConfig]:
            """Generate k_split configs based on input matrix dimensions."""
            from smith._inductor.utils import get_k_splits

            m, k = fake_tensors["a"].shape[-2:]
            _, n = fake_tensors["b"].shape[-2:]

            k_splits_list = get_k_splits(m, n, k)

            return [CustomOpConfig(k_splits=k) for k in k_splits_list]

        register_custom_op_autotuning(
            matmul_relu_epilogue_dynamic_op,
            config_generator=generate_k_split_configs,
            name="matmul_relu_epilogue_dynamic_autotuned",
            input_gen_fns={
                "a": lambda fake_tensor: smith.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
                "b": lambda fake_tensor: smith.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
                "bias": lambda fake_tensor: smith.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
            },
        )

        # Test multiple shapes to verify dynamic config generation
        test_shapes = [
            (256, 16384, 1024),
            (256, 65536, 1024),
        ]

        for m, k, n in test_shapes:
            # Use helper function to create test inputs
            a, b, bias = self._create_decompose_k_inputs(m, k, n)

            @smith.compile
            def test_model(a, b, bias):
                return matmul_relu_epilogue_dynamic_op(a, b, bias)

            smith._dynamo.reset()

            with config.patch(
                max_autotune=True,
                benchmark_fusion=True,
            ):
                compiled_result = test_model(a, b, bias)

            def reference_model(a, b, bias):
                matmul_result = a @ b
                biased = matmul_result + bias
                activated = smith.relu(biased)
                scaled = activated * 2.0
                return scaled

            expected = reference_model(a, b, bias)

            smith.testing.assert_close(
                compiled_result,
                expected,
                rtol=2e-1,
                atol=5e-1,
                msg=f"Failed for shape ({m}, {k}, {n})",
            )

    @skipIfXpu
    def test_multi_parameter_tuning(self):
        """Test autotuning with multiple parameters for combinatorial parameter exploration.

        Validates parametric tuning with multiple parameters (scale_mode and chunk_size)
        to test combinatorial exploration of the parameter space.
        """
        test_op_name = f"test_lib::multi_param_{id(self)}"

        def multi_param_scaling(
            x: smith.Tensor,
            factor: smith.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ) -> smith.Tensor:
            """Different scaling approaches controlled by scale_mode parameter."""
            if scale_mode == 1:
                # Simple broadcasting
                return x * factor
            elif scale_mode == 2:
                # Process in chunks
                batch_size, seq_len = x.shape[:2]
                chunks = []
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    chunk = x[:, start:end]
                    chunks.append(chunk * factor)
                return smith.cat(chunks, dim=1)
            elif scale_mode == 3:
                # Using einsum for scaling
                return smith.einsum("...i,i->...i", x, factor)

        @smith.library.custom_op(test_op_name, mutates_args=())
        def multi_param_op(
            x: smith.Tensor,
            factor: smith.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ) -> smith.Tensor:
            return multi_param_scaling(x, factor, scale_mode, chunk_size)

        @multi_param_op.register_fake
        def _(
            x: smith.Tensor,
            factor: smith.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ):
            return smith.empty_like(x)

        # Use explicit configs with scale_mode and chunk_size parameters as tuning knobs
        register_custom_op_autotuning(
            multi_param_op,
            configs=[
                CustomOpConfig(scale_mode=1),  # Broadcast
                CustomOpConfig(scale_mode=2, chunk_size=16),  # Chunked 16
                CustomOpConfig(scale_mode=2, chunk_size=32),  # Chunked 32
                CustomOpConfig(scale_mode=3),  # Einsum
            ],
            name="multi_param_autotuned",
            input_gen_fns={
                "x": lambda t: smith.randn_like(t, device=self.device) * 0.1,
                "factor": lambda t: smith.ones(
                    t.shape[-1], device=self.device, dtype=t.dtype
                ),
            },
        )

        # Create test inputs
        test_x = smith.randn(4, 64, 128, device=self.device, dtype=self.dtype)
        test_factor = smith.ones(128, device=self.device, dtype=self.dtype) * 2.0

        # Verify numerical equivalence across all approaches
        expected_result = test_x * test_factor

        # Test each scale_mode variant
        configs = [
            (1, 16),  # broadcast, chunk_size ignored
            (2, 16),  # chunked with size 16
            (2, 32),  # chunked with size 32
            (3, 16),  # einsum, chunk_size ignored
        ]

        for scale_mode, chunk_size in configs:
            result = multi_param_scaling(
                test_x, test_factor, scale_mode=scale_mode, chunk_size=chunk_size
            )
            smith.testing.assert_close(
                result,
                expected_result,
                rtol=1e-5,
                atol=1e-5,
                msg=f"scale_mode {scale_mode} with chunk_size {chunk_size} not equivalent to expected",
            )

        # Test autotuning
        self._run_autotune_test(
            multi_param_op, (test_x, test_factor), expected_result, "MultiParam"
        )

    @skipIfXpu
    def test_range_based_static_shape_no_cond_dispatch(self):
        """Test dispatch code generation for static vs dynamic shapes.

        Static shapes (dynamic=False): No dispatch logic, best impl is inlined.
        Dynamic shapes (dynamic=True): Dispatch logic generated for runtime selection.
        """
        import re

        from smith._inductor.utils import run_and_get_code

        test_op_name = f"test_lib::static_no_cond_{id(self)}"

        def impl_a(x: smith.Tensor, weight: smith.Tensor) -> smith.Tensor:
            return x + weight

        def impl_b(x: smith.Tensor, weight: smith.Tensor) -> smith.Tensor:
            return x + weight.view(1, 1, -1).expand_as(x)

        def impl_c(x: smith.Tensor, weight: smith.Tensor) -> smith.Tensor:
            return smith.add(x, weight)

        @smith.library.custom_op(test_op_name, mutates_args=())
        def static_no_cond_op(x: smith.Tensor, weight: smith.Tensor) -> smith.Tensor:
            return x + weight

        @static_no_cond_op.register_fake
        def _(x: smith.Tensor, weight: smith.Tensor):
            return smith.empty_like(x)

        register_custom_op_autotuning(
            static_no_cond_op,
            configs=[
                CustomOpConfig(impl_a),
                CustomOpConfig(impl_b),
                CustomOpConfig(impl_c),
            ],
            dispatch_on={"tensor_name": "x", "dim": 1},
            split_points=[64, 128, 256, 512],
            input_gen_fns={
                "x": lambda fake: smith.randn_like(fake, device=self.device),
                "weight": lambda fake: smith.randn_like(fake, device=self.device),
            },
        )

        autotune_backends = "TRITON" if self.device == "cuda" else "ATEN"
        test_x = smith.randn(2, 96, 32, device=self.device, dtype=self.dtype)
        test_weight = smith.randn(32, device=self.device, dtype=self.dtype)

        def find_shape_dispatch(code_list):
            pattern = re.compile(r"if\s+s\d+\s*[<>=]")
            return [
                line.strip()
                for code in code_list
                for line in code.split("\n")
                if pattern.search(line)
            ]

        def test_model(x, weight):
            return static_no_cond_op(x, weight)

        # Static shape (dynamic=False) - should NOT have shape dispatch
        smith._dynamo.reset()
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends=autotune_backends,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            result_static, code_list_static = run_and_get_code(
                smith.compile(test_model, dynamic=False), test_x, test_weight
            )
            self.assertEqual(result_static, test_x + test_weight)

        dispatch_static = find_shape_dispatch(code_list_static)
        if dispatch_static:
            print(f"[Static] Found dispatch logic: {dispatch_static}")
        else:
            print("[Static] No dispatch logic found (expected for static shapes)")
        self.assertFalse(
            dispatch_static, "Static shapes should not have dispatch logic"
        )

        # Dynamic shape (dynamic=True) - SHOULD have shape dispatch
        smith._dynamo.reset()
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends=autotune_backends,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            result_dynamic, code_list_dynamic = run_and_get_code(
                smith.compile(test_model, dynamic=True), test_x, test_weight
            )
            self.assertEqual(result_dynamic, test_x + test_weight)

        dispatch_dynamic = find_shape_dispatch(code_list_dynamic)
        if dispatch_dynamic:
            print(f"[Dynamic] Found dispatch logic: {dispatch_dynamic}")
        else:
            print("[Dynamic] No dispatch logic found (unexpected for dynamic shapes)")
        self.assertTrue(dispatch_dynamic, "Dynamic shapes should have dispatch logic")


if __name__ == "__main__":
    run_tests()
