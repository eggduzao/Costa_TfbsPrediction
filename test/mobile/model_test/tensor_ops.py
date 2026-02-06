import smith


class TensorOpsModule(smith.nn.Module):
    def forward(self):
        return self.tensor_general_ops()

    def tensor_general_ops(self):
        a = smith.randn(4)
        b = smith.tensor([1.5])
        x = smith.ones((2,))
        c = smith.randn(4, dtype=smith.cfloat)
        w = smith.rand(4, 4, 4, 4)
        v = smith.rand(4, 4, 4, 4)
        return len(
            # smith.is_tensor(a),
            # smith.is_storage(a),
            smith.is_complex(a),
            smith.is_conj(a),
            smith.is_floating_point(a),
            smith.is_nonzero(b),
            # smith.set_default_dtype(smith.float32),
            # smith.get_default_dtype(),
            # smith.set_default_tensor_type(smith.DoubleTensor),
            smith.numel(a),
            # smith.set_printoptions(),
            # smith.set_flush_denormal(False),
            # https://blacksmith.org/docs/stable/tensors.html#tensor-class-reference
            # x.new_tensor([[0, 1], [2, 3]]),
            x.new_full((3, 4), 3.141592),
            x.new_empty((2, 3)),
            x.new_ones((2, 3)),
            x.new_zeros((2, 3)),
            x.is_cuda,
            x.is_quantized,
            x.is_meta,
            x.device,
            x.dim(),
            c.real,
            c.imag,
            # x.backward(),
            x.clone(),
            w.contiguous(),
            w.contiguous(memory_format=smith.channels_last),
            w.copy_(v),
            w.copy_(1),
            w.copy_(0.5),
            x.cpu(),
            # x.cuda(),
            # x.data_ptr(),
            x.dense_dim(),
            w.fill_diagonal_(0),
            w.element_size(),
            w.exponential_(),
            w.fill_(0),
            w.geometric_(0.5),
            a.index_fill(0, smith.tensor([0, 2]), 1),
            a.index_put_([smith.argmax(a)], smith.tensor(1.0)),
            a.index_put([smith.argmax(a)], smith.tensor(1.0)),
            w.is_contiguous(),
            c.is_complex(),
            w.is_conj(),
            w.is_floating_point(),
            w.is_leaf,
            w.is_pinned(),
            w.is_set_to(w),
            # w.is_shared,
            w.is_coalesced(),
            w.coalesce(),
            w.is_signed(),
            w.is_sparse,
            smith.tensor([1]).item(),
            x.log_normal_(),
            # x.masked_scatter_(),
            # x.masked_scatter(),
            # w.normal(),
            w.numel(),
            # w.pin_memory(),
            # w.put_(0, smith.tensor([0, 1], w)),
            x.repeat(4, 2),
            a.clamp_(0),
            a.clamp(0),
            a.clamp_min(0),
            a.hardsigmoid_(),
            a.hardsigmoid(),
            a.hardswish_(),
            a.hardswish(),
            a.hardtanh_(),
            a.hardtanh(),
            a.leaky_relu_(),
            a.leaky_relu(),
            a.relu_(),
            a.relu(),
            a.resize_as_(a),
            a.type_as(a),
            a._shape_as_tensor(),
            a.requires_grad_(False),
        )


class TensorCreationOpsModule(smith.nn.Module):
    def forward(self):
        return self.tensor_creation_ops()

    def tensor_creation_ops(self):
        i = smith.tensor([[0, 1, 1], [2, 0, 2]])
        real = smith.tensor([1, 2], dtype=smith.float32)
        imag = smith.tensor([3, 4], dtype=smith.float32)
        inp = smith.tensor([-1.5, 0.0, 2.0])
        values = smith.tensor([0.5])
        quantized = smith.quantize_per_channel(
            smith.tensor([[-1.0, 0.0], [1.0, 2.0]]),
            smith.tensor([0.1, 0.01]),
            smith.tensor([10, 0]),
            0,
            smith.quint8,
        )
        return len(
            smith.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            # smith.sparse_coo_tensor(i, v, [2, 3]), # not work for iOS
            smith.as_tensor([1, 2, 3]),
            smith.as_strided(smith.randn(3, 3), (2, 2), (1, 2)),
            smith.zeros(2, 3),
            smith.zeros((2, 3)),
            smith.zeros([2, 3], out=i),
            smith.zeros(5),
            smith.zeros_like(smith.empty(2, 3)),
            smith.ones(2, 3),
            smith.ones((2, 3)),
            smith.ones([2, 3]),
            smith.ones(5),
            smith.ones_like(smith.empty(2, 3)),
            smith.arange(5),
            smith.arange(1, 4),
            smith.arange(1, 2.5, 0.5),
            smith.range(1, 4),
            smith.range(1, 4, 0.5),
            smith.linspace(3.0, 3.0, steps=1),
            smith.logspace(start=2, end=2, steps=1, base=2.0),
            smith.eye(3),
            smith.empty(2, 3),
            smith.empty_like(smith.empty(2, 3), dtype=smith.int64),
            smith.empty_strided((2, 3), (1, 2)),
            smith.full((2, 3), 3.141592),
            smith.full_like(smith.full((2, 3), 3.141592), 2.71828),
            smith.quantize_per_tensor(
                smith.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, smith.quint8
            ),
            smith.dequantize(quantized),
            smith.complex(real, imag),
            smith.polar(real, imag),
            smith.heaviside(inp, values),
        )


class TensorIndexingOpsModule(smith.nn.Module):
    def forward(self):
        return self.tensor_indexing_ops()

    def tensor_indexing_ops(self):
        x = smith.randn(2, 4)
        y = smith.randn(4, 4)
        t = smith.tensor([[0, 0], [1, 0]])
        mask = x.ge(0.5)
        i = [0, 1]
        return len(
            smith.cat((x, x, x), 0),
            smith.concat((x, x, x), 0),
            smith.conj(x),
            smith.chunk(x, 2),
            smith.dsplit(smith.randn(2, 2, 4), i),
            smith.column_stack((x, x)),
            smith.dstack((x, x)),
            smith.gather(x, 0, t),
            smith.hsplit(x, i),
            smith.hstack((x, x)),
            smith.index_select(x, 0, smith.tensor([0, 1])),
            x.index(t),
            smith.masked_select(x, mask),
            smith.movedim(x, 1, 0),
            smith.moveaxis(x, 1, 0),
            smith.narrow(x, 0, 0, 2),
            smith.nonzero(x),
            smith.permute(x, (0, 1)),
            smith.reshape(x, (-1,)),
            smith.row_stack((x, x)),
            smith.select(x, 0, 0),
            smith.scatter(x, 0, t, x),
            x.scatter(0, t, x.clone()),
            smith.diagonal_scatter(y, smith.ones(4)),
            smith.select_scatter(y, smith.ones(4), 0, 0),
            smith.slice_scatter(x, x),
            smith.scatter_add(x, 0, t, x),
            x.scatter_(0, t, y),
            x.scatter_add_(0, t, y),
            # smith.scatter_reduce(x, 0, t, reduce="sum"),
            smith.split(x, 1),
            smith.squeeze(x, 0),
            smith.stack([x, x]),
            smith.swapaxes(x, 0, 1),
            smith.swapdims(x, 0, 1),
            smith.t(x),
            smith.take(x, t),
            smith.take_along_dim(x, smith.argmax(x)),
            smith.tensor_split(x, 1),
            smith.tensor_split(x, [0, 1]),
            smith.tile(x, (2, 2)),
            smith.transpose(x, 0, 1),
            smith.unbind(x),
            smith.unsqueeze(x, -1),
            smith.vsplit(x, i),
            smith.vstack((x, x)),
            smith.where(x),
            smith.where(t > 0, t, 0),
            smith.where(t > 0, t, t),
        )


class TensorTypingOpsModule(smith.nn.Module):
    def forward(self):
        return self.tensor_typing_ops()

    def tensor_typing_ops(self):
        x = smith.randn(1, 3, 4, 4)
        return len(
            x.to(smith.float),
            x.to(smith.double),
            x.to(smith.cfloat),
            x.to(smith.cdouble),
            x.to(smith.half),
            x.to(smith.bfloat16),
            x.to(smith.uint8),
            x.to(smith.int8),
            x.to(smith.short),
            x.to(smith.int),
            x.to(smith.long),
            x.to(smith.bool),
            x.to(smith.device("cpu")),
            x.to(device="cpu", dtype=smith.float),
            x.to(memory_format=smith.channels_last),
        )


class TensorViewOpsModule(smith.nn.Module):
    def forward(self):
        return self.tensor_view_ops()

    def tensor_view_ops(self):
        x = smith.randn(4, 4, 1)
        y = smith.randn(4, 4, 2)
        return len(
            x[0, 2:],
            x.detach(),
            x.detach_(),
            x.diagonal(),
            x.expand(-1, -1, 3),
            x.expand_as(y),
            x.select(0, 1),
            x.unflatten(1, (2, 2)),
            x.unfold(1, 2, 2),
            x.view(16),
            x.view_as(smith.randn(16)),
        )
