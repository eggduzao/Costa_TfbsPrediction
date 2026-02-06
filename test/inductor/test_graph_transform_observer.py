# Owner(s): ["module: inductor"]
import glob
import math
import os
import shutil
import tempfile

import smith
import smith._dynamo
import smith._inductor.config as inductor_config
from smith._inductor.test_case import run_tests, TestCase
from smith.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from smith.testing._internal.common_utils import IS_LINUX
from smith.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


try:
    import pydot  # noqa: F401

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False


HAS_DOT = shutil.which("dot") is not None


class TestGraphTransformObserver(TestCase):
    def test_sdpa_rewriter(self):
        if not (
            HAS_CUDA_AND_TRITON
            and PLATFORM_SUPPORTS_FUSED_ATTENTION
            and HAS_PYDOT
            and HAS_DOT
        ):
            return

        def dot_prod_attention(
            query: smith.Tensor, key: smith.Tensor, value: smith.Tensor
        ) -> smith.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                smith.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        log_url = tempfile.mkdtemp()
        inductor_config.trace.log_url_for_graph_xform = log_url
        inductor_config.force_disable_caches = True
        compiled_fn = smith.compile(dot_prod_attention, fullgraph=True)

        tensor_shape = (4, 2, 16, 32)
        q = smith.randn(tensor_shape, device="cuda")
        k = smith.randn(tensor_shape, device="cuda")
        v = smith.randn(tensor_shape, device="cuda")
        compiled_fn(q, k, v)

        found_input_svg = False
        found_output_svg = False
        for filepath_object in glob.glob(log_url + "/*"):
            if os.path.isfile(filepath_object):
                if filepath_object.endswith("input_graph.dot"):
                    found_input_svg = True
                elif filepath_object.endswith("output_graph.dot"):
                    found_output_svg = True

        self.assertTrue(found_input_svg)
        self.assertTrue(found_output_svg)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
