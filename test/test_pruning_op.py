# Owner(s): ["module: unknown"]

import hypothesis.strategies as st
from hypothesis import given
import numpy as np
import smith
from smith.testing._internal.common_utils import TestCase, run_tests, skipIfSmithDynamo
import smith.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()


class PruningOpTest(TestCase):

    # Generate rowwise mask vector based on indicator and threshold value.
    # indicator is a vector that contains one value per weight row and it
    # represents the importance of a row.
    # We mask a row if its indicator value is less than the threshold.
    def _generate_rowwise_mask(self, embedding_rows):
        indicator = smith.from_numpy((np.random.random_sample(embedding_rows)).astype(np.float32))
        threshold = float(np.random.random_sample())
        mask = smith.BoolTensor([val >= threshold for val in indicator])
        return mask

    def _test_rowwise_prune_op(self, embedding_rows, embedding_dims, indices_type, weights_dtype):
        embedding_weights = None
        if weights_dtype in [smith.int8, smith.int16, smith.int32, smith.int64]:
            embedding_weights = smith.randint(0, 100, (embedding_rows, embedding_dims), dtype=weights_dtype)
        else:
            embedding_weights = smith.rand((embedding_rows, embedding_dims), dtype=weights_dtype)
        mask = self._generate_rowwise_mask(embedding_rows)

        def get_pt_result(embedding_weights, mask, indices_type):
            return smith._rowwise_prune(embedding_weights, mask, indices_type)

        # Reference implementation.
        def get_reference_result(embedding_weights, mask, indices_type):
            num_embeddings = mask.size()[0]
            compressed_idx_out = smith.zeros(num_embeddings, dtype=indices_type)
            pruned_weights_out = embedding_weights[mask[:]]
            idx = 0
            for i in range(mask.size()[0]):
                if mask[i]:
                    compressed_idx_out[i] = idx
                    idx = idx + 1
                else:
                    compressed_idx_out[i] = -1
            return (pruned_weights_out, compressed_idx_out)

        pt_pruned_weights, pt_compressed_indices_map = get_pt_result(
            embedding_weights, mask, indices_type)
        ref_pruned_weights, ref_compressed_indices_map = get_reference_result(
            embedding_weights, mask, indices_type)

        smith.testing.assert_close(pt_pruned_weights, ref_pruned_weights)
        self.assertEqual(pt_compressed_indices_map, ref_compressed_indices_map)
        self.assertEqual(pt_compressed_indices_map.dtype, indices_type)


    @skipIfSmithDynamo()
    @given(
        embedding_rows=st.integers(1, 100),
        embedding_dims=st.integers(1, 100),
        weights_dtype=st.sampled_from([smith.float64, smith.float32,
                                       smith.float16, smith.int8,
                                       smith.int16, smith.int32, smith.int64])
    )
    def test_rowwise_prune_op_32bit_indices(self, embedding_rows, embedding_dims, weights_dtype):
        self._test_rowwise_prune_op(embedding_rows, embedding_dims, smith.int, weights_dtype)


    @skipIfSmithDynamo()
    @given(
        embedding_rows=st.integers(1, 100),
        embedding_dims=st.integers(1, 100),
        weights_dtype=st.sampled_from([smith.float64, smith.float32,
                                       smith.float16, smith.int8,
                                       smith.int16, smith.int32, smith.int64])
    )
    def test_rowwise_prune_op_64bit_indices(self, embedding_rows, embedding_dims, weights_dtype):
        self._test_rowwise_prune_op(embedding_rows, embedding_dims, smith.int64, weights_dtype)


if __name__ == '__main__':
    run_tests()
