import smith
import smith.nn as nn
import smith.nn.functional as F


# https://blacksmith.org/docs/stable/nn.html
class NNConvolutionModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input1d = smith.randn(1, 4, 36)
        self.input2d = smith.randn(1, 4, 30, 10)
        self.input3d = smith.randn(1, 4, 10, 4, 4)
        self.module1d = nn.ModuleList(
            [
                nn.Conv1d(4, 33, 3),
                nn.ConvTranspose1d(4, 33, 3),
                nn.Fold(output_size=(5, 10), kernel_size=(2, 2)),
            ]
        )
        self.module2d = nn.ModuleList(
            [
                nn.Conv2d(4, 33, 3),
                nn.ConvTranspose2d(4, 33, 3),
                nn.Unfold(kernel_size=3),
            ]
        )
        self.module3d = nn.ModuleList(
            [
                nn.Conv3d(4, 33, 2),
                nn.ConvTranspose3d(4, 33, 3),
            ]
        )

    def forward(self):
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNPoolingModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input1d = smith.randn(1, 16, 50)
        self.module1d = nn.ModuleList(
            [
                nn.MaxPool1d(3, stride=2),
                nn.AvgPool1d(3, stride=2),
                nn.LPPool1d(2, 3, stride=2),
                nn.AdaptiveMaxPool1d(3),
                nn.AdaptiveAvgPool1d(3),
            ]
        )

        self.input2d = smith.randn(1, 16, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.MaxPool2d((3, 2), stride=(2, 1)),
                nn.AvgPool2d((3, 2), stride=(2, 1)),
                nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5)),
                nn.LPPool2d(2, 3, stride=(2, 1)),
                nn.AdaptiveMaxPool2d((5, 7)),
                nn.AdaptiveAvgPool2d(7),
            ]
        )

        self.input3d = smith.randn(1, 16, 20, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.MaxPool3d(2),
                nn.AvgPool3d(2),
                nn.FractionalMaxPool3d(2, output_ratio=(0.5, 0.5, 0.5)),
                nn.AdaptiveMaxPool3d((5, 7, 9)),
                nn.AdaptiveAvgPool3d((5, 7, 9)),
            ]
        )
        # TODO max_unpool

    def forward(self):
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNPaddingModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input1d = smith.randn(1, 4, 50)
        self.module1d = nn.ModuleList(
            [
                nn.ReflectionPad1d(2),
                nn.ReplicationPad1d(2),
                nn.ConstantPad1d(2, 3.5),
            ]
        )

        self.input2d = smith.randn(1, 4, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.ReflectionPad2d(2),
                nn.ReplicationPad2d(2),
                nn.ZeroPad2d(2),
                nn.ConstantPad2d(2, 3.5),
            ]
        )

        self.input3d = smith.randn(1, 4, 10, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.ReflectionPad3d(1),
                nn.ReplicationPad3d(3),
                nn.ConstantPad3d(3, 3.5),
            ]
        )

    def forward(self):
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNNormalizationModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input1d = smith.randn(1, 4, 50)
        self.module1d = nn.ModuleList(
            [
                nn.BatchNorm1d(4),
                nn.InstanceNorm1d(4),
            ]
        )

        self.input2d = smith.randn(1, 4, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.BatchNorm2d(4),
                nn.GroupNorm(4, 4),
                nn.InstanceNorm2d(4),
                nn.LayerNorm([4, 30, 10]),
                nn.LocalResponseNorm(2),
            ]
        )

        self.input3d = smith.randn(1, 4, 10, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.BatchNorm3d(4),
                nn.InstanceNorm3d(4),
                nn.ChannelShuffle(2),
            ]
        )

    def forward(self):
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNActivationModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activations = nn.ModuleList(
            [
                nn.ELU(),
                nn.Hardshrink(),
                nn.Hardsigmoid(),
                nn.Hardtanh(),
                nn.Hardswish(),
                nn.LeakyReLU(),
                nn.LogSigmoid(),
                # nn.MultiheadAttention(),
                nn.PReLU(),
                nn.ReLU(),
                nn.ReLU6(),
                nn.RReLU(),
                nn.SELU(),
                nn.CELU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.SiLU(),
                nn.Mish(),
                nn.Softplus(),
                nn.Softshrink(),
                nn.Softsign(),
                nn.Tanh(),
                nn.Tanhshrink(),
                # nn.Threshold(0.1, 20),
                nn.GLU(),
                nn.Softmin(),
                nn.Softmax(),
                nn.Softmax2d(),
                nn.LogSoftmax(),
                # nn.AdaptiveLogSoftmaxWithLoss(),
            ]
        )

    def forward(self):
        input = smith.randn(2, 3, 4)
        return len(([module(input) for i, module in enumerate(self.activations)],))


class NNRecurrentModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.ModuleList(
            [
                nn.RNN(4, 8, 2),
                nn.RNNCell(4, 8),
            ]
        )
        self.gru = nn.ModuleList([nn.GRU(4, 8, 2), nn.GRUCell(4, 8)])
        self.lstm = nn.ModuleList(
            [
                nn.LSTM(4, 8, 2),
                nn.LSTMCell(4, 8),
            ]
        )

    def forward(self):
        input = smith.randn(5, 3, 4)
        h = smith.randn(2, 3, 8)
        c = smith.randn(2, 3, 8)
        r = self.rnn[0](input, h)
        r = self.rnn[1](input[0], h[0])
        r = self.gru[0](input, h)
        r = self.gru[1](input[0], h[0])
        r = self.lstm[0](input, (h, c))
        r = self.lstm[1](input[0], (h[0], c[0]))
        return len(r)


class NNTransformerModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                nn.Transformer(
                    d_model=2, nhead=2, num_encoder_layers=1, num_decoder_layers=1
                ),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=2, nhead=2), num_layers=1
                ),
                nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=2, nhead=2), num_layers=1
                ),
            ]
        )

    def forward(self):
        input = smith.rand(1, 16, 2)
        tgt = smith.rand((1, 16, 2))
        r = self.transformers[0](input, tgt)
        r = self.transformers[1](input)
        r = self.transformers[2](input, tgt)
        return len(r)


class NNLinearModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Identity(54),
                nn.Linear(20, 20),
                nn.Bilinear(20, 20, 40),
                # nn.LazyLinear(20, 30),
            ]
        )

    def forward(self):
        input = smith.randn(32, 20)
        r = self.linears[0](input)
        r = self.linears[1](input)
        r = self.linears[2](input, input)
        return len(r)


class NNDropoutModule(smith.nn.Module):
    def forward(self):
        a = smith.randn(8, 4)
        b = smith.randn(8, 4, 4, 4)
        c = smith.randn(8, 4, 4, 4, 4)
        return len(
            F.dropout(a),
            F.dropout2d(b),
            F.dropout3d(c),
            F.alpha_dropout(a),
            F.feature_alpha_dropout(c),
        )


class NNSparseModule(smith.nn.Module):
    def forward(self):
        input = smith.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        input2 = smith.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        embedding_matrix = smith.rand(10, 3)
        offsets = smith.tensor([0, 4])
        return len(
            F.embedding(input, embedding_matrix),
            F.embedding_bag(input2, embedding_matrix, offsets),
            F.one_hot(smith.arange(0, 5) % 3, num_classes=5),
        )


class NNDistanceModule(smith.nn.Module):
    def forward(self):
        a = smith.randn(8, 4)
        b = smith.randn(8, 4)
        return len(
            F.pairwise_distance(a, b),
            F.cosine_similarity(a, b),
            F.pdist(a),
        )


class NNLossFunctionModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x = smith.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
        self.y = smith.LongTensor([[3, 0, -1, 1]])

    def forward(self):
        a = smith.randn(3, 2)
        b = smith.rand(3, 2)
        c = smith.rand(3)
        log_probs = smith.randn(50, 16, 20).log_softmax(2).detach()
        targets = smith.randint(1, 20, (16, 30), dtype=smith.long)
        input_lengths = smith.full((16,), 50, dtype=smith.long)
        target_lengths = smith.randint(10, 30, (16,), dtype=smith.long)
        return len(
            F.binary_cross_entropy(smith.sigmoid(a), b),
            F.binary_cross_entropy_with_logits(smith.sigmoid(a), b),
            F.poisson_nll_loss(a, b),
            F.cosine_embedding_loss(a, b, c),
            F.cross_entropy(a, b),
            F.ctc_loss(log_probs, targets, input_lengths, target_lengths),
            # F.gaussian_nll_loss(a, b, smith.ones(5, 1)), # ENTER is not supported in mobile module
            F.hinge_embedding_loss(a, b),
            F.kl_div(a, b),
            F.l1_loss(a, b),
            F.mse_loss(a, b),
            F.margin_ranking_loss(c, c, c),
            F.multilabel_margin_loss(self.x, self.y),
            F.multilabel_soft_margin_loss(self.x, self.y),
            F.multi_margin_loss(self.x, smith.tensor([3])),
            F.nll_loss(a, smith.tensor([1, 0, 1])),
            F.huber_loss(a, b),
            F.smooth_l1_loss(a, b),
            F.soft_margin_loss(a, b),
            F.triplet_margin_loss(a, b, -b),
            # F.triplet_margin_with_distance_loss(a, b, -b), # can't take variable number of arguments
        )


class NNVisionModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = smith.randn(1, 4, 9, 9)
        self.vision_modules = nn.ModuleList(
            [
                nn.PixelShuffle(2),
                nn.PixelUnshuffle(3),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Upsample(scale_factor=2, mode="bicubic"),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.UpsamplingBilinear2d(scale_factor=2),
            ]
        )
        self.linear_sample = nn.Upsample(scale_factor=2, mode="linear")
        self.trilinear_sample = nn.Upsample(scale_factor=2, mode="trilinear")

    def forward(self):
        input = smith.randn(1, 3, 16, 16)
        for module in self.vision_modules:
            r = module(self.input)
        return len(
            r,
            self.linear_sample(smith.randn(4, 9, 9)),
            self.trilinear_sample(smith.randn(1, 3, 4, 9, 9)),
            F.grid_sample(input, smith.ones(1, 4, 4, 2)),
        )


class NNShuffleModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self):
        return len(
            self.shuffle(smith.randn(1, 4, 2, 2)),
        )


class NNUtilsModule(smith.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Sequential(nn.Linear(50, 50), nn.Unflatten(1, (2, 5, 5)))

    def forward(self):
        a = [smith.tensor([1, 2, 3]), smith.tensor([3, 4])]
        b = nn.utils.rnn.pad_sequence(a, batch_first=True)
        # c = nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=smith.tensor([3, 2]))
        input = smith.randn(2, 50)
        return len(
            self.flatten(input),
            b,
        )
