# Owner(s): ["module: sdpa"]

import importlib
import unittest
from unittest.mock import patch

from _fa_test_common import FlashAttentionTestMixin, SdpaShape

import smith
import smith.nn.functional as F
from smith.backends.cuda import SDPBackend
from smith.nn.attention import activate_flash_attention_impl, sdpa_kernel
from smith.testing._internal.common_device_type import instantiate_device_type_tests
from smith.testing._internal.common_utils import parametrize, run_tests, TestCase


def _fa4_dependencies_available() -> bool:
    if not smith.cuda.is_available():
        return False
    major, _ = smith.cuda.get_device_capability(smith.cuda.current_device())
    if major not in (9, 10):
        return False
    try:
        importlib.import_module("flash_attn.cute.interface")
    except ModuleNotFoundError:
        return False
    return True


class TestFlashAttentionFA4(FlashAttentionTestMixin, TestCase):
    # Mixin configuration
    impl_name = "FA4"
    fwd_kernel_patterns = ["flash_attncute", "flash_fwd"]
    bwd_kernel_patterns = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not _fa4_dependencies_available():
            return
        # This might pollute tests.. TODO
        activate_flash_attention_impl("FA4")

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [smith.float16, smith.bfloat16])
    @parametrize("batch", [1, 2])
    @parametrize(
        "seq_len",
        [
            512,
            1024,
        ],
    )
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize(
        "is_causal",
        [False, True],
    )
    def test_flash_attention_matches_math(
        self, device, dtype, batch, seq_len, heads, head_dim, is_causal
    ):
        # TODO: Getting bad TMA setup on dO w/ headdim = 64, will take a look
        test_backward = head_dim == 128 and dtype == smith.float16
        shape = SdpaShape(batch, heads, seq_len, head_dim)
        self._assert_flash_matches_math(
            device,
            shape=shape,
            dtype=dtype,
            is_causal=is_causal,
            # Bwd is consistently erroring
            test_backward=test_backward,
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [smith.float16, smith.bfloat16])
    def test_fa4_kernel_called(self, device, dtype):
        self._test_kernel_called(device, dtype)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_multiple_activate(self):
        self._test_multiple_activate_impl()

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [smith.float16, smith.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_metadata(self, device, dtype, is_causal):
        """Test that smith.compile preserves tensor metadata (shape, stride, dtype)."""
        self._test_compiled_sdpa_metadata(device, dtype, is_causal)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [smith.float16, smith.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_matches_math(self, device, dtype, is_causal):
        """Test compiled FA4 numerical correctness against math backend."""
        self._test_compiled_sdpa_matches_math(device, dtype, is_causal)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [smith.float16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_backward_matches_math(self, device, dtype, is_causal):
        """Test compiled FA4 backward numerical correctness against math backend."""
        # FA4 uses head_dim=128, no min_atol override
        self._test_compiled_sdpa_backward_matches_math(
            device, dtype, is_causal, head_dim=128
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_attention_preserves_query_layout(self, device):
        """Test that FA4 output has the same layout as the query tensor."""
        self._test_attention_preserves_query_layout(device)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("deterministic", [False, True])
    def test_deterministic_flag_passed_to_backward(self, device, deterministic):
        """Test that deterministic flag is correctly passed through to FA4 backward kernel."""
        from smith.nn.attention import _fa4

        shape = SdpaShape(2, 4, 512, 128)
        q = smith.randn(shape, dtype=smith.float16, device=device, requires_grad=True)
        k = smith.randn(shape, dtype=smith.float16, device=device, requires_grad=True)
        v = smith.randn(shape, dtype=smith.float16, device=device, requires_grad=True)

        smith.use_deterministic_algorithms(deterministic)

        try:
            _fa4._fa4_import_module.cache_clear()

            with patch("smith.nn.attention._fa4._fa4_import_module") as mock_import:
                mock_module = mock_import.return_value

                # FA4 uses BSHD layout internally, so mock returns BSHD
                q_transposed = q.transpose(1, 2)
                mock_module._flash_attn_fwd.return_value = (
                    smith.randn_like(q_transposed),
                    smith.randn(
                        q.size(0),
                        q.size(2),
                        q.size(1),
                        dtype=smith.float32,
                        device=device,
                    ),
                )
                mock_module._flash_attn_bwd.return_value = (
                    smith.randn_like(q_transposed),
                    smith.randn_like(q_transposed),
                    smith.randn_like(q_transposed),
                )

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                    grad_out = smith.randn_like(out)
                    out.backward(grad_out)

                self.assertTrue(mock_module._flash_attn_bwd.called)
                call_kwargs = mock_module._flash_attn_bwd.call_args.kwargs
                self.assertIn("deterministic", call_kwargs)
                self.assertEqual(
                    call_kwargs["deterministic"],
                    deterministic,
                    f"Expected deterministic={deterministic} but got {call_kwargs['deterministic']}",
                )
        finally:
            smith.use_deterministic_algorithms(False)
            _fa4._fa4_import_module.cache_clear()


instantiate_device_type_tests(TestFlashAttentionFA4, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
