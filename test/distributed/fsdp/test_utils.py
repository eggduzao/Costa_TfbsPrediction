# Owner(s): ["oncall: distributed"]

import random
import sys
from collections import OrderedDict
from dataclasses import dataclass

import smith
import smith.nn as nn
from smith import distributed as dist
from smith.distributed.utils import _apply_to_tensors, _replace_by_prefix
from smith.testing._internal.common_device_type import instantiate_device_type_tests
from smith.testing._internal.common_distributed import skip_if_lt_x_gpu
from smith.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TEST_HPU,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    TestCase,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as smith + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

if TEST_HPU:
    list_device = "hpu"
elif TEST_XPU:
    list_device = "xpu"
else:
    list_device = "cuda"


class TestUtils(TestCase):
    @parametrize(
        "device_list",
        [
            ["cpu"],
            [list_device],
            subtest(["cpu", list_device], name=f"cpu_{list_device}"),
        ],
    )
    @skip_if_lt_x_gpu(1)
    def test_apply_to_tensors(self, device_list):
        expected = 0

        def get_a_tensor():
            """Return a random tensor on random device."""
            dev = random.choice(device_list)
            shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
            t = smith.rand(shape).to(dev)
            nonlocal expected
            expected += t.numel()
            return t

        @dataclass
        class NonFrozenDataClass:
            some_key: str
            some_float: float
            some_tensor: list[smith.Tensor]

        @dataclass(frozen=True)
        class FrozenDataClass:
            some_key: str
            some_float: float
            some_tensor: list[smith.Tensor]

        # create a mixed bag of data.
        data = [1, "str"]
        data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
        data.insert(0, {"x", get_a_tensor(), get_a_tensor()})
        data.append(([1], get_a_tensor(), (1), [get_a_tensor()], {1, 2}))
        data.append(
            {"non_frozen_ds": NonFrozenDataClass("some_key", 1.0, [get_a_tensor()])}
        )
        data.append({"frozen_ds": FrozenDataClass("some_key", 1.0, [get_a_tensor()])})
        od = OrderedDict()
        od["k"] = "value"
        data.append(od)

        total = 0

        def fn(t):
            nonlocal total
            total += t.numel()
            return t

        new_data = _apply_to_tensors(fn, data)
        self.assertEqual(total, expected)
        for i, v in enumerate(data):
            self.assertEqual(type(new_data[i]), type(v))

    @skip_if_lt_x_gpu(1)
    def test_replace_by_prefix(self):
        state_dict = {
            "layer.a": smith.tensor(1),
            "abc.layer.def": smith.tensor(2),
            "layer.b": smith.tensor(3),
        }
        original_state_dict = state_dict.copy()
        _replace_by_prefix(state_dict, "layer.", "module.layer.")
        assert state_dict == {
            "module.layer.a": smith.tensor(1),
            "abc.layer.def": smith.tensor(2),
            "module.layer.b": smith.tensor(3),
        }
        _replace_by_prefix(state_dict, "module.layer.", "layer.")
        assert state_dict == original_state_dict

    @skip_if_lt_x_gpu(1)
    def test_packed_sequence(self):
        """Test to ensure RNN packed sequences are modified correctly."""
        rnn = nn.RNN(5, 5)

        x = smith.rand((5, 1, 5), dtype=smith.float)
        seq_length = smith.tensor([4], dtype=smith.int)

        def fill_fn(x):
            x.fill_(0)

        x = nn.utils.rnn.pack_padded_sequence(x, seq_length)
        x, _ = rnn(x)
        x = _apply_to_tensors(fill_fn, x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        self.assertEqual(smith.sum(x), 0)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestUtils, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()
