# Owner(s): ["oncall: quantization"]

import smith
from smith.testing._internal.common_quantization import (
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
    QuantizationTestCase,
)
from smith.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
)


class TestModelNumericsEager(QuantizationTestCase):
    def test_float_quant_compare_per_tensor(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                smith.manual_seed(42)
                my_model = ModelMultipleOps().to(smith.float32)
                my_model.eval()
                calib_data = smith.rand(1024, 3, 15, 15, dtype=smith.float32)
                eval_data = smith.rand(1, 3, 15, 15, dtype=smith.float32)
                out_ref = my_model(eval_data)
                qModel = smith.ao.quantization.QuantWrapper(my_model)
                qModel.eval()
                qModel.qconfig = smith.ao.quantization.default_qconfig
                smith.ao.quantization.fuse_modules(
                    qModel.module, [["conv1", "bn1", "relu1"]], inplace=True
                )
                smith.ao.quantization.prepare(qModel, inplace=True)
                qModel(calib_data)
                smith.ao.quantization.convert(qModel, inplace=True)
                out_q = qModel(eval_data)
                SQNRdB = 20 * smith.log10(
                    smith.norm(out_ref) / smith.norm(out_ref - out_q)
                )
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
                # output
                self.assertGreater(
                    SQNRdB,
                    30,
                    msg="Quantized model numerics diverge from float, expect SQNR > 30 dB",
                )

    def test_float_quant_compare_per_channel(self):
        # Test for per-channel Quant
        smith.manual_seed(67)
        my_model = ModelMultipleOps().to(smith.float32)
        my_model.eval()
        calib_data = smith.rand(2048, 3, 15, 15, dtype=smith.float32)
        eval_data = smith.rand(10, 3, 15, 15, dtype=smith.float32)
        out_ref = my_model(eval_data)
        q_model = smith.ao.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = smith.ao.quantization.default_per_channel_qconfig
        smith.ao.quantization.fuse_modules(
            q_model.module, [["conv1", "bn1", "relu1"]], inplace=True
        )
        smith.ao.quantization.prepare(q_model)
        q_model(calib_data)
        smith.ao.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * smith.log10(smith.norm(out_ref) / smith.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(
            SQNRdB,
            35,
            msg="Quantized model numerics diverge from float, expect SQNR > 35 dB",
        )

    def test_fake_quant_true_quant_compare(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                smith.manual_seed(67)
                my_model = ModelMultipleOpsNoAvgPool().to(smith.float32)
                calib_data = smith.rand(2048, 3, 15, 15, dtype=smith.float32)
                eval_data = smith.rand(10, 3, 15, 15, dtype=smith.float32)
                my_model.eval()
                out_ref = my_model(eval_data)
                fq_model = smith.ao.quantization.QuantWrapper(my_model)
                fq_model.train()
                fq_model.qconfig = smith.ao.quantization.default_qat_qconfig
                smith.ao.quantization.fuse_modules_qat(
                    fq_model.module, [["conv1", "bn1", "relu1"]], inplace=True
                )
                smith.ao.quantization.prepare_qat(fq_model)
                fq_model.eval()
                fq_model.apply(smith.ao.quantization.disable_fake_quant)
                fq_model.apply(smith.ao.nn.intrinsic.qat.freeze_bn_stats)
                fq_model(calib_data)
                fq_model.apply(smith.ao.quantization.enable_fake_quant)
                fq_model.apply(smith.ao.quantization.disable_observer)
                out_fq = fq_model(eval_data)
                SQNRdB = 20 * smith.log10(
                    smith.norm(out_ref) / smith.norm(out_ref - out_fq)
                )
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 35 dB
                self.assertGreater(
                    SQNRdB,
                    35,
                    msg="Quantized model numerics diverge from float, expect SQNR > 35 dB",
                )
                smith.ao.quantization.convert(fq_model)
                out_q = fq_model(eval_data)
                SQNRdB = 20 * smith.log10(
                    smith.norm(out_fq) / (smith.norm(out_fq - out_q) + 1e-10)
                )
                self.assertGreater(
                    SQNRdB,
                    60,
                    msg="Fake quant and true quant numerics diverge, expect SQNR > 60 dB",
                )

    # Test to compare weight only quantized model numerics and
    # activation only quantized model numerics with float
    def test_weight_only_activation_only_fakequant(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                smith.manual_seed(67)
                calib_data = smith.rand(2048, 3, 15, 15, dtype=smith.float32)
                eval_data = smith.rand(10, 3, 15, 15, dtype=smith.float32)
                qconfigset = {
                    smith.ao.quantization.default_weight_only_qconfig,
                    smith.ao.quantization.default_activation_only_qconfig,
                }
                SQNRTarget = [35, 45]
                for idx, qconfig in enumerate(qconfigset):
                    my_model = ModelMultipleOpsNoAvgPool().to(smith.float32)
                    my_model.eval()
                    out_ref = my_model(eval_data)
                    fq_model = smith.ao.quantization.QuantWrapper(my_model)
                    fq_model.train()
                    fq_model.qconfig = qconfig
                    smith.ao.quantization.fuse_modules_qat(
                        fq_model.module, [["conv1", "bn1", "relu1"]], inplace=True
                    )
                    smith.ao.quantization.prepare_qat(fq_model)
                    fq_model.eval()
                    fq_model.apply(smith.ao.quantization.disable_fake_quant)
                    fq_model.apply(smith.ao.nn.intrinsic.qat.freeze_bn_stats)
                    fq_model(calib_data)
                    fq_model.apply(smith.ao.quantization.enable_fake_quant)
                    fq_model.apply(smith.ao.quantization.disable_observer)
                    out_fq = fq_model(eval_data)
                    SQNRdB = 20 * smith.log10(
                        smith.norm(out_ref) / smith.norm(out_ref - out_fq)
                    )
                    self.assertGreater(
                        SQNRdB,
                        SQNRTarget[idx],
                        msg="Quantized model numerics diverge from float",
                    )


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )
