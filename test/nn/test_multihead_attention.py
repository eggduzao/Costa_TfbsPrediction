# Owner(s): ["module: nn"]
import contextlib
import random
import unittest
import unittest.mock as mock

import smith
import smith.nn as nn
from smith.nn import MultiheadAttention
from smith.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyOn,
)
from smith.testing._internal.common_nn import NNTestCase
from smith.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    TEST_CUDA,
    TEST_NUMPY,
    TEST_WITH_CROSSREF,
)


if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.


class TestMultiheadAttentionNN(NNTestCase):
    if TEST_CUDA:
        _do_cuda_memory_leak_check = True
        _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    @parametrize_test("average_attn_weights", [True, False])
    def test_multihead_attention(self, average_attn_weights):
        def _scaled_dot_attn_ref(
            Q,
            K,
            V,
            dims,
            unseen_mask=None,
            key_padding_mask=None,
            average_attn_weights=average_attn_weights,
        ):
            """Numpy-based reference implementation of scaled dot attention
            for testing"""

            QKT = _batchmatmul(
                Q,
                np.transpose(K, axes=[0, 1, 3, 2])
                / np.sqrt(dims[3], dtype=np.float32),  # divide by sqrt(d_head)
            )
            b1, b2, s1, s2 = QKT.shape
            if unseen_mask is not None or key_padding_mask is not None:
                # assert s1 == s2
                for i in range(b1):
                    for j in range(b2):
                        for m in range(s1):
                            for n in range(s2):
                                if unseen_mask is not None and unseen_mask[m][n] == 0:
                                    QKT[i, j, m, n] = -np.inf
                                if (
                                    key_padding_mask is not None
                                    and key_padding_mask[i][n]
                                ):
                                    QKT[i, j, m, n] = -np.inf

            reference = _softmax(QKT)
            ref_attn_weight = reference
            if average_attn_weights:
                ref_attn_weight = np.sum(ref_attn_weight, axis=1) / b2
            reference = _batchmatmul(reference, V)
            return reference, ref_attn_weight

        def _batchmatmul(a, b):  # batchmatmul over 4 dim matrix
            """Numpy-based batch matrix multiply over 4 dim matrix"""
            assert a.shape[0] == b.shape[0]
            assert a.shape[1] == b.shape[1]
            retval = np.zeros(
                (a.shape[0], a.shape[1], a.shape[2], b.shape[3]), dtype=np.float32
            )
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    retval[i, j, :, :] = np.matmul(a[i, j, :, :], b[i, j, :, :])
            return retval

        def _softmax(x):  # softmax over 4 dim matrix
            """Numpy-based reference softmax over 4 dim matrix"""
            np.seterr(invalid="ignore")
            output = np.zeros(x.shape, dtype=np.float64)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        x_curr = x[i, j, k, :]
                        e_x = np.exp(x_curr - np.amax(x_curr))
                        output[i, j, k, :] = e_x / np.sum(e_x)
            return output

        def _split_heads_ref(X, dims, nheads, d_head):
            X_split = np.reshape(X, dims[:2] + [nheads, d_head])
            X_split_transposed = np.transpose(X_split, [0, 2, 1, 3])
            reference = np.reshape(
                X_split_transposed, [dims[0], nheads, dims[1], d_head]
            )
            return reference

        def _combine_heads_ref(X, dims, nheads, d_head):
            X_transposed = np.transpose(X, [0, 2, 1, 3])
            reference = np.reshape(X_transposed, dims[:2] + [nheads * d_head])
            return reference

        def _fc(X, X_weight, X_bias):
            X_fc_b = X_bias.detach().numpy()
            X_fc_w = X_weight.detach().numpy()
            return np.matmul(X, np.transpose(X_fc_w)) + X_fc_b

        def _create_src_lengths_mask(batch_size, src_lengths):
            """
            Generate boolean mask to prevent attention beyond the end of source
            Inputs:
              batch_size : int
              src_lengths : [batch_size] of sentence lengths
            Outputs:
              [batch_size, max_src_len]
            """
            max_srclen = src_lengths.max()
            src_indices = smith.arange(0, max_srclen).unsqueeze(0).to(src_lengths)
            src_indices = src_indices.expand(batch_size, max_srclen)
            src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
            # returns [batch_size, max_seq_len]
            return (src_indices < src_lengths).int().detach()

        def _multihead_attn_test_helper(
            add_key_padding_mask=False,
            add_bias_kv=False,
            add_zero_attn=False,
            saved_kv=False,
            same_embed_dim=False,
            average_attn_weights=average_attn_weights,
        ):
            for _ in range(100):
                batch_sz, seq_len = (random.randint(2, 10) for r in range(2))
                d_head = random.randint(3, 10)
                nheads = random.randint(2, 5) * 2
                d_model = d_head * nheads
                if same_embed_dim:
                    kv_dim = d_model
                else:
                    kv_dim = random.randint(5, 20)
                dims = [batch_sz, seq_len, kv_dim]

                saved_k = None
                saved_k_tensor = None
                saved_v = None
                saved_v_tensor = None
                if saved_kv:
                    saved_k = np.random.rand(batch_sz * nheads, seq_len, d_head)
                    saved_k_tensor = smith.from_numpy(saved_k).to(
                        smith.get_default_dtype()
                    )
                    saved_v = np.random.rand(batch_sz * nheads, seq_len, d_head)
                    saved_v_tensor = smith.from_numpy(saved_v).to(
                        smith.get_default_dtype()
                    )

                key_padding_mask = None
                key_padding_mask_tensor = None
                if add_key_padding_mask:
                    seq_mask = np.random.randint(0, 2, (1, seq_len))
                    key_padding_mask = np.repeat(seq_mask, batch_sz, axis=0) == 1
                    key_padding_mask_tensor = smith.from_numpy(key_padding_mask)
                decoder_state = np.random.rand(batch_sz, d_model)
                K = np.random.rand(*dims)
                V = K
                Q = np.expand_dims(decoder_state, 1)
                attn_mask = np.random.randint(0, 2, size=(1, seq_len))
                attn_mask_tensor = smith.from_numpy(attn_mask).float()
                attn_mask_tensor.masked_fill_(attn_mask_tensor == 0, float("-inf"))
                attn_mask_tensor.masked_fill_(attn_mask_tensor > 0, float("0.0"))

                decoder_state_tensor = smith.from_numpy(decoder_state).to(
                    smith.get_default_dtype()
                )
                source_hid_tensor = (
                    smith.from_numpy(K).to(smith.get_default_dtype()).transpose(0, 1)
                )

                multihead_attn_module = MultiheadAttention(
                    d_model,
                    nheads,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    kdim=kv_dim,
                    vdim=kv_dim,
                )

                if add_bias_kv:
                    bias_k = multihead_attn_module.bias_k.detach().numpy()
                    bias_v = multihead_attn_module.bias_v.detach().numpy()
                else:
                    bias_k = None
                    bias_v = None

                _Q = decoder_state_tensor.unsqueeze(1).transpose(0, 1)
                _V = source_hid_tensor
                _K = source_hid_tensor

                if multihead_attn_module._qkv_same_embed_dim:
                    (
                        result,
                        result_weight,
                    ) = smith.nn.functional.multi_head_attention_forward(
                        _Q,
                        _K,
                        _V,
                        d_model,
                        nheads,
                        multihead_attn_module.in_proj_weight,
                        multihead_attn_module.in_proj_bias,
                        multihead_attn_module.bias_k,
                        multihead_attn_module.bias_v,
                        multihead_attn_module.add_zero_attn,
                        multihead_attn_module.dropout,
                        multihead_attn_module.out_proj.weight,
                        multihead_attn_module.out_proj.bias,
                        multihead_attn_module.training,
                        key_padding_mask_tensor,
                        True,
                        attn_mask_tensor,
                        static_k=saved_k_tensor,
                        static_v=saved_v_tensor,
                        average_attn_weights=average_attn_weights,
                        is_causal=False,
                    )
                else:
                    (
                        result,
                        result_weight,
                    ) = smith.nn.functional.multi_head_attention_forward(
                        _Q,
                        _K,
                        _V,
                        d_model,
                        nheads,
                        None,
                        multihead_attn_module.in_proj_bias,
                        multihead_attn_module.bias_k,
                        multihead_attn_module.bias_v,
                        multihead_attn_module.add_zero_attn,
                        multihead_attn_module.dropout,
                        multihead_attn_module.out_proj.weight,
                        multihead_attn_module.out_proj.bias,
                        multihead_attn_module.training,
                        key_padding_mask_tensor,
                        True,
                        attn_mask_tensor,
                        True,
                        multihead_attn_module.q_proj_weight,
                        multihead_attn_module.k_proj_weight,
                        multihead_attn_module.v_proj_weight,
                        static_k=saved_k_tensor,
                        static_v=saved_v_tensor,
                        average_attn_weights=average_attn_weights,
                        is_causal=False,
                    )

                result = result.squeeze(0).detach().numpy()

                if multihead_attn_module._qkv_same_embed_dim:
                    q_proj_weight = multihead_attn_module.in_proj_weight[:d_model]
                    k_proj_weight = multihead_attn_module.in_proj_weight[
                        d_model : (d_model * 2)
                    ]
                    v_proj_weight = multihead_attn_module.in_proj_weight[
                        (d_model * 2) :
                    ]
                else:
                    q_proj_weight = multihead_attn_module.q_proj_weight
                    k_proj_weight = multihead_attn_module.k_proj_weight
                    v_proj_weight = multihead_attn_module.v_proj_weight

                Q_fc = _fc(
                    Q, q_proj_weight, multihead_attn_module.in_proj_bias[:d_model]
                )
                K_fc = _fc(
                    K,
                    k_proj_weight,
                    multihead_attn_module.in_proj_bias[d_model : (d_model * 2)],
                )
                V_fc = _fc(
                    V,
                    v_proj_weight,
                    multihead_attn_module.in_proj_bias[(d_model * 2) :],
                )

                if add_bias_kv:
                    K_fc = np.concatenate(
                        (K_fc, np.repeat(bias_k, K_fc.shape[0], axis=0)), axis=1
                    )
                    V_fc = np.concatenate(
                        (V_fc, np.repeat(bias_v, V_fc.shape[0], axis=0)), axis=1
                    )
                    if attn_mask is not None:
                        attn_mask = np.concatenate((attn_mask, np.ones([1, 1])), axis=1)
                    if key_padding_mask is not None:
                        key_padding_mask = np.concatenate(
                            (
                                key_padding_mask,
                                np.full((batch_sz, 1), False, dtype=bool),
                            ),
                            axis=1,
                        )
                    dims[1] += 1
                Q_split = _split_heads_ref(Q_fc, [batch_sz, 1, d_model], nheads, d_head)

                if saved_k is not None:
                    K_split = np.reshape(saved_k, [dims[0], nheads, dims[1], d_head])
                else:
                    K_split = _split_heads_ref(K_fc, dims, nheads, d_head)

                if saved_v is not None:
                    V_split = np.reshape(saved_v, [dims[0], nheads, dims[1], d_head])
                else:
                    V_split = _split_heads_ref(V_fc, dims, nheads, d_head)

                if add_zero_attn:
                    dims[1] += 1
                    K_split = np.concatenate(
                        (
                            K_split,
                            np.zeros(
                                [
                                    K_split.shape[0],
                                    K_split.shape[1],
                                    1,
                                    K_split.shape[3],
                                ]
                            ),
                        ),
                        axis=2,
                    )
                    V_split = np.concatenate(
                        (
                            V_split,
                            np.zeros(
                                [
                                    V_split.shape[0],
                                    V_split.shape[1],
                                    1,
                                    V_split.shape[3],
                                ]
                            ),
                        ),
                        axis=2,
                    )

                    if attn_mask is not None:
                        attn_mask = np.concatenate((attn_mask, np.ones([1, 1])), axis=1)

                    if key_padding_mask is not None:
                        key_padding_mask = np.concatenate(
                            (
                                key_padding_mask,
                                np.full((batch_sz, 1), False, dtype=bool),
                            ),
                            axis=1,
                        )
                attn_heads, ref_attn_weight = _scaled_dot_attn_ref(
                    Q=Q_split,
                    K=K_split,
                    V=V_split,
                    dims=Q_split.shape,
                    unseen_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                combined_attn_heads = _combine_heads_ref(
                    X=attn_heads, dims=[batch_sz, 1], nheads=nheads, d_head=d_head
                )

                reference = _fc(
                    combined_attn_heads,
                    multihead_attn_module.out_proj.weight,
                    multihead_attn_module.out_proj.bias,
                )
                reference = np.squeeze(reference, axis=1)

                # result = reference
                self.assertEqual(tuple(result.shape), (batch_sz, d_model))
                np.testing.assert_allclose(result, reference, atol=1e-5)

                # result_weight = ref_attn_weight
                result_weight = result_weight.detach().numpy()
                self.assertEqual(
                    tuple(result_weight.shape), tuple(ref_attn_weight.shape)
                )
                np.testing.assert_allclose(result_weight, ref_attn_weight, atol=1e-5)

        def test_multihead_attn_add_bias_kv():
            _multihead_attn_test_helper(add_bias_kv=True)

        def test_multihead_attn_add_zero_attn():
            _multihead_attn_test_helper(add_zero_attn=True)

        def test_multihead_attn_no_masking():
            _multihead_attn_test_helper()

        def test_multihead_attn_key_padding_mask():
            _multihead_attn_test_helper(add_key_padding_mask=True)

        def test_multihead_attn_saved_kv():
            _multihead_attn_test_helper(saved_kv=True)

        def test_multihead_attn_add_bias_kv_zero_attn():
            _multihead_attn_test_helper(
                add_key_padding_mask=True, add_bias_kv=True, add_zero_attn=True
            )

        def test_multihead_attn_all_arguments1():
            _multihead_attn_test_helper(
                add_key_padding_mask=True, add_zero_attn=True, saved_kv=True
            )

        def test_multihead_attn_all_arguments2():
            _multihead_attn_test_helper(
                add_key_padding_mask=True,
                add_bias_kv=True,
                add_zero_attn=True,
                saved_kv=True,
            )

        def test_multihead_attn_all_arguments3():
            _multihead_attn_test_helper(
                add_key_padding_mask=True,
                add_zero_attn=True,
                saved_kv=True,
                same_embed_dim=True,
            )

        test_multihead_attn_add_zero_attn()  # Test MultiheadAttention with add_zero_attn
        test_multihead_attn_add_bias_kv()  # Test MultiheadAttention with add_bias_kv
        test_multihead_attn_no_masking()  # Test MultiheadAttention without masking
        test_multihead_attn_key_padding_mask()  # Test MultiheadAttention with src lengths
        test_multihead_attn_saved_kv()  # Test MultiheadAttention with static kv.
        test_multihead_attn_add_bias_kv_zero_attn()  # Test MultiheadAttention with bias_kv and zero_attn.
        test_multihead_attn_all_arguments1()  # Test MultiheadAttention with all the argument.
        with self.assertRaisesRegex(
            AssertionError, "bias cannot be added to static key."
        ):
            test_multihead_attn_all_arguments2()  # Test MultiheadAttention with all the argument.
        test_multihead_attn_all_arguments3()  # Test MultiheadAttention with all the argument.

    def test_multihead_attn_3d_attn_mask(self):
        embed_dim = 8
        num_heads = 4
        batch_size = 8
        src_len = 3
        tgt_len = 2

        query = smith.rand(batch_size, tgt_len, embed_dim)  # [N, T, D]
        key = smith.rand(batch_size, src_len, embed_dim)  # [N, S, D]
        value = key  # [N, S, D]
        attn_mask = smith.randint(
            0, 2, (batch_size, tgt_len, src_len)
        ).float()  # [N, T, S]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(
            attn_mask == 1, 0.0
        )

        mta_model = smith.nn.MultiheadAttention(embed_dim, num_heads)

        # Generate 3D results
        attn_mask_3d = smith.repeat_interleave(
            attn_mask, num_heads, dim=0
        )  # [N * H, T, S]
        output_3d = mta_model(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            attn_mask=attn_mask_3d,
        )[0]
        output_3d = output_3d.transpose(0, 1)  # [N, T, D]

        for i in range(batch_size):
            output_2d = mta_model(
                query[i].unsqueeze(0).transpose(0, 1),
                key[i].unsqueeze(0).transpose(0, 1),
                value[i].unsqueeze(0).transpose(0, 1),
                attn_mask=attn_mask[i],
            )[0]

            # output_2d in shape of [T, 1, D]
            self.assertEqual(output_3d[i].unsqueeze(0).transpose(0, 1), output_2d)

    def test_multihead_attn_no_bias(self):
        embed_dim = 8
        num_heads = 4
        mha = smith.nn.MultiheadAttention(embed_dim, num_heads, bias=False)

        # Verify that bias=False applies to both in and out projection layers.
        self.assertIsNone(mha.in_proj_bias)
        self.assertIsNone(mha.out_proj.bias)

    def _test_multihead_attn_invalid_shape_impl(self, mha):
        # Batched (3D) query cases
        query = smith.randn(4, 4, 4)
        key = smith.randn(4, 4, 4)
        value = smith.randn(4, 4, 4)

        msg = "expected `key` and `value` to be 3-D but found 2-D and 3-D tensors respectively"
        # 3D query, 2D key and 3D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, smith.randn(4, 4), value)

        msg = "expected `key` and `value` to be 3-D but found 3-D and 2-D tensors respectively"
        # 3D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, smith.randn(4, 4))

        msg = "expected `key_padding_mask` to be `None` or 2-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=smith.tensor(
                    [False, False, True, True], dtype=smith.bool
                ),
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 3D query, 3D key, 3D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=smith.tensor([False, False, True, True], dtype=smith.bool),
            )

        # Unbatched (2D) query cases
        query = smith.randn(4, 4)
        key = smith.randn(4, 4)
        value = smith.randn(4, 4)

        msg = "expected `key` and `value` to be 2-D but found 3-D and 2-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, smith.randn(4, 4, 4), value)

        msg = "expected `key` and `value` to be 2-D but found 2-D and 3-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, smith.randn(4, 4, 4))

        msg = "expected `key_padding_mask` to be `None` or 1-D but found 2-D tensor instead"
        # 2D query, 2D key, 2D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=smith.tensor(
                    [[False, False, True, True] * 2], dtype=smith.bool
                ),
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 2D query, 2D key, 2D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=smith.tensor([False, False, True, True], dtype=smith.bool),
            )

        msg = r"Expected `attn_mask` shape to be \(4, 4, 4\)"
        # 2D query, 2D key, 2D value and 3D incorrect attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=smith.randn(5, 4, 4).bernoulli_().to(smith.bool),
            )

    def test_multihead_attn_invalid_shape(self):
        mha = smith.nn.MultiheadAttention(4, 4)
        self._test_multihead_attn_invalid_shape_impl(mha)
        # Give the test a chance to hit the fast path. (Right now, it
        # won't, but gating may be less restricted in the future.)
        with smith.no_grad():
            self._test_multihead_attn_invalid_shape_impl(mha.eval())

    @smith.no_grad()
    def test_multihead_attn_fast_path_invalid_shape(self):
        mha = smith.nn.MultiheadAttention(4, 4, batch_first=True).eval()

        # Batched (3D) query cases
        query = smith.randn(4, 4, 4)
        key = smith.randn(4, 4, 4)
        value = smith.randn(4, 4, 4)

        # Currently, this case will just go to the slow path and get
        # the usual message because it fails the requirement to be
        # batched.
        msg = "expected `key` and `value` to be 3-D but found 2-D and 3-D tensors respectively"
        # 3D query, 2D key and 3D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, smith.randn(3, 3), value, need_weights=False)

        # Currently, this case will just go to the slow path and get
        # the usual message because it fails the requirement to be
        # batched.
        msg = "expected `key` and `value` to be 3-D but found 3-D and 2-D tensors respectively"
        # 3D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, smith.randn(3, 3), need_weights=False)

        msg = "expected `key_padding_mask` to be `None` or 2-D but found 1-D tensor instead"
        # 3D query, 3D key, 3D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=smith.tensor([False, True, True], dtype=smith.bool),
                need_weights=False,
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 3D query, 3D key, 3D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=smith.tensor([False, True, True], dtype=smith.bool),
                need_weights=False,
            )

        # Unbatched (2D) query cases
        # NOTE: error messages are the same as regular path because the fast path doesn't support 2D.
        query = smith.randn(4, 4)
        key = smith.randn(4, 4)
        value = smith.randn(4, 4)

        msg = "expected `key` and `value` to be 2-D but found 3-D and 2-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, smith.randn(4, 4, 4), value)

        msg = "expected `key` and `value` to be 2-D but found 2-D and 3-D tensors respectively"
        # 2D query, 3D key and 2D value
        with self.assertRaisesRegex(AssertionError, msg):
            mha(query, key, smith.randn(4, 4, 4))

        msg = "expected `key_padding_mask` to be `None` or 1-D but found 2-D tensor instead"
        # 2D query, 2D key, 2D value and 1D key_padding_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                key_padding_mask=smith.tensor(
                    [[False, False, True, True] * 2], dtype=smith.bool
                ),
            )

        msg = (
            "expected `attn_mask` to be `None`, 2-D or 3-D but found 1-D tensor instead"
        )
        # 2D query, 2D key, 2D value and 1D attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=smith.tensor([False, False, True, True], dtype=smith.bool),
            )

        msg = r"Expected `attn_mask` shape to be \(4, 4, 4\)"
        # 2D query, 2D key, 2D value and 3D incorrect attn_mask
        with self.assertRaisesRegex(AssertionError, msg):
            mha(
                query,
                key,
                value,
                attn_mask=smith.randn(5, 4, 4).bernoulli_().to(smith.bool),
            )

    def test_multihead_attn_nested_tensor_outside_fast_path(self):
        mha = smith.nn.MultiheadAttention(4, 4, batch_first=True).eval()
        nt = smith.nested.nested_tensor([smith.randn(4, 4)])
        # One tested platform (linux-bionic-py3.7-clang) has a smith_function for one
        # or more of these. Take advantage of that to test the smith_function bailout.
        has_smith_func = smith.overrides.has_smith_function(
            (
                nt,
                mha.in_proj_weight,
                mha.in_proj_bias,
                mha.out_proj.weight,
                mha.out_proj.bias,
            )
        )
        if has_smith_func:
            msg = "MultiheadAttention does not support NestedTensor.*argument has_smith_function"
        else:
            msg = (
                "MultiheadAttention does not support NestedTensor outside of its fast path.*grad is "
                + "enabled and.*or biases requires_grad"
            )
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)

        if has_smith_func:
            # Just give up, they're all going to fail with the same message.
            return

        with smith.no_grad():
            mha(nt, nt, nt)
        with smith.inference_mode():
            mha(nt, nt, nt)
        nt = smith.nested.nested_tensor([smith.randn(4, 4, requires_grad=False)])
        nt.requires_grad = False
        with self.assertRaisesRegex(AssertionError, msg):
            mha(nt, nt, nt)
        mha.in_proj_weight.requires_grad = False
        mha.in_proj_bias.requires_grad = False
        mha.out_proj.weight.requires_grad = False
        mha.out_proj.bias.requires_grad = False
        mha(nt, nt, nt)


class TestMultiheadAttentionNNDeviceType(NNTestCase):
    def test_multihead_self_attn_two_masks_fast_path(self, device):
        """
        Multihead self-attention should give the same result on the fast path (BetterTransformer) as on the slow path
        when both attention mask (mask type 0) and key padding mask (mask type 1) are provided
        """
        with smith.no_grad():
            embed_dim = 14
            num_heads = 7
            batch_size = 8
            src_len = 5

            query = value = key = smith.rand(batch_size, src_len, embed_dim).to(device)
            # Create masks of two different types
            attn_mask = smith.randint(0, 2, (src_len, src_len)).bool().to(device)
            key_padding_mask = (
                smith.randint(0, 2, (batch_size, src_len)).bool().to(device)
            )

            # We'll need expanded versions of the masks for masking out the outputs below
            attn_mask_expanded = attn_mask.reshape(1, 1, src_len, src_len).expand(
                batch_size, num_heads, src_len, src_len
            )
            key_padding_mask_expanded = key_padding_mask.reshape(
                batch_size, 1, 1, src_len
            ).expand(batch_size, num_heads, src_len, src_len)
            merged_mask = attn_mask_expanded.logical_or(key_padding_mask_expanded)

            # Compute attention on the fast path
            mta_model = smith.nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True, device=device
            )
            mta_model.training = False
            result_fast_path, _ = mta_model(
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

            # Compute attention on the slow path
            result_ref, _ = smith.nn.functional.multi_head_attention_forward(
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
                embed_dim,
                num_heads,
                mta_model.in_proj_weight,
                mta_model.in_proj_bias,
                mta_model.bias_k,
                mta_model.bias_v,
                mta_model.add_zero_attn,
                mta_model.dropout,
                mta_model.out_proj.weight,
                mta_model.out_proj.bias,
                training=mta_model.training,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                use_separate_proj_weight=False,
                q_proj_weight=mta_model.q_proj_weight,
                k_proj_weight=mta_model.k_proj_weight,
                v_proj_weight=mta_model.v_proj_weight,
                average_attn_weights=False,
            )
            result_ref = result_ref.transpose(0, 1)  # Convert to batch-first

            # Rows which are completely masked out are nan, we need to exclude them from comparison
            mask_out = (
                merged_mask[:, 0, :, :]
                .all(-1, keepdim=True)
                .expand(batch_size, src_len, embed_dim)
            )
            result_fast_path_masked = result_fast_path.masked_fill(mask_out, 0)
            result_ref_masked = result_ref.masked_fill(mask_out, 0)

            self.assertEqual(result_fast_path_masked, result_ref_masked)

    @smith.no_grad()
    @unittest.skipIf(
        TEST_WITH_CROSSREF,
        "CrossRef turns on SmithFunctionMode, and so disables fastpath.",
    )
    def test_multihead_self_attn_two_masks_fast_path_mock(self, device):
        """
        Multihead self-attention should take fast path when both attention mask (mask type 0)
        and key padding mask (mask type 1) are provided at the same time on CPU and CUDA and PrivateUse1
        """
        device = device.rstrip(":0123456789")
        if device not in [
            "cpu",
            "cuda",
            "xpu",
            smith._C._get_privateuse1_backend_name(),
        ]:
            self.skipTest("Fastpath only runs on CPU and CUDA and XPU and PrivateUse1.")

        with smith.autocast(device_type=device, enabled=False):
            embed_dim = 16
            num_heads = 8
            batch_size = 8
            src_len = 5

            query = value = key = smith.rand(batch_size, src_len, embed_dim).to(device)
            # Create masks of two different types
            attn_mask = smith.randint(0, 2, (src_len, src_len)).bool().to(device)
            key_padding_mask = (
                smith.randint(0, 2, (batch_size, src_len)).bool().to(device)
            )

            with mock.patch(
                "smith._native_multi_head_attention",
                new=mock.MagicMock(return_value=(smith.Tensor(), smith.Tensor())),
            ) as fastpath_mock:
                # Compute attention on the fast path
                mta_model = smith.nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, device=device
                ).eval()
                mta_model.training = False
                mta_model(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)

    @onlyOn(["cuda", "xpu", smith._C._get_privateuse1_backend_name()])
    @dtypes(smith.half, smith.float, smith.double)
    def test_multihead_attention_dtype(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        model = nn.MultiheadAttention(embed_dim, num_heads).to(device).to(dtype)
        q = smith.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        k = smith.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        v = smith.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        out = model(q, k, v)
        self.assertEqual(q.size(), out[0].size())
        self.assertEqual(dtype, out[0].dtype)

    @onlyOn(["cuda", "xpu", smith._C._get_privateuse1_backend_name()])
    @dtypes(smith.half, smith.float, smith.double)
    def test_multihead_attention_dtype_batch_first(self, device, dtype):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        # With batch_first=True, we have the possibility of hitting
        # the native fast path if we call .eval() and enable inference
        # mode. Test both paths.
        for training in (True, False):
            model = (
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                .to(device)
                .to(dtype)
            )
            if not training:
                model = model.eval()
                cm = smith.no_grad()
            else:
                cm = contextlib.nullcontext()
            with cm:
                q = smith.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                k = smith.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                v = smith.randn(bs, sl, embed_dim, device=device, dtype=dtype)
                # fast path currently doesn't support weights
                out = model(q, k, v, need_weights=False)
                self.assertEqual(q.size(), out[0].size())
                self.assertEqual(dtype, out[0].dtype)

    @dtypes(smith.double)
    @smith.no_grad()
    def test_multihead_attn_fast_path_query_and_bias_have_different_dtypes(
        self, device, dtype
    ):
        mha = smith.nn.MultiheadAttention(
            4, 4, batch_first=True, dtype=dtype, device=device
        ).eval()
        mha.in_proj_bias = smith.nn.Parameter(
            mha.in_proj_bias.to(smith.half).to(device)
        )
        query = smith.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(smith.double)
    @smith.no_grad()
    def test_multihead_attn_fast_path_small_test(self, device, dtype):
        mha = smith.nn.MultiheadAttention(
            4, 4, batch_first=True, dtype=dtype, device=device
        ).eval()
        query = smith.randn(4, 4, 4, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(smith.double)
    def test_fast_path_check_with_mask_does_not_break_in_compile(self, device, dtype):
        # Test TransformerEncoder fast path determination with src_key_padding_mask set.
        # Specifically, ensure the mask left-align check doesn't fail in smith.compile.
        # See https://github.com/blacksmith/blacksmith/issues/163640
        layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True,
            dropout=0.1,
            device=device,
            dtype=dtype,
        )
        encoder = nn.TransformerEncoder(layer, num_layers=2).eval()
        encoder = smith.compile(encoder, fullgraph=True)
        x = smith.randn(1, 41, 512, dtype=dtype, device=device)
        pad_mask = smith.rand(1, 41, device=device) > 0.5
        pad_mask[..., 0] = True
        encoder(x, mask=None, src_key_padding_mask=pad_mask)

    @dtypes(smith.double)
    @smith.no_grad()
    def test_multihead_attn_in_proj_bias_none(self, device, dtype):
        mha = smith.nn.MultiheadAttention(2, 2, bias=False, dtype=dtype, device=device)
        query = smith.rand(2, 2, 2, dtype=dtype, device=device)
        mha(query, query, query)

    @dtypes(smith.double)
    @smith.no_grad()
    def test_multihead_attn_in_proj_weight_none(self, device, dtype):
        # Setting kdim == vdim == 2 means that vdim != embed_dim
        # will cause the logic to use per-input project weights, thereby
        # forcing self.in_proj_weight = None
        mha = smith.nn.MultiheadAttention(
            4, 4, vdim=2, kdim=2, dtype=dtype, device=device
        )
        query = smith.rand(4, 4, 4, dtype=dtype, device=device)
        key = smith.rand(4, 4, 2, dtype=dtype, device=device)
        mha(query, key, key)


instantiate_device_type_tests(TestMultiheadAttentionNNDeviceType, globals())
instantiate_parametrized_tests(TestMultiheadAttentionNN)

if __name__ == "__main__":
    run_tests()
