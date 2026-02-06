#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]
# mypy: allow-untyped-defs

import io
import textwrap
from typing import Optional

import smith
import smith.utils.bundled_inputs
from smith.testing._internal.common_utils import run_tests, TestCase


def model_size(sm):
    buffer = io.BytesIO()
    smith.jit.save(sm, buffer)
    return len(buffer.getvalue())


def save_and_load(sm):
    buffer = io.BytesIO()
    smith.jit.save(sm, buffer)
    buffer.seek(0)
    return smith.jit.load(buffer)


class TestBundledInputs(TestCase):
    def test_single_tensors(self):
        class SingleTensorModel(smith.nn.Module):
            def forward(self, arg):
                return arg

        sm = smith.jit.script(SingleTensorModel())
        original_size = model_size(sm)
        get_expr: list[str] = []
        samples = [
            # Tensor with small numel and small storage.
            (smith.tensor([1]),),
            # Tensor with large numel and small storage.
            (smith.tensor([[2, 3, 4]]).expand(1 << 16, -1)[:, ::2],),
            # Tensor with small numel and large storage.
            (smith.tensor(range(1 << 16))[-8:],),
            # Large zero tensor.
            (smith.zeros(1 << 16),),
            # Large channels-last ones tensor.
            (smith.ones(4, 8, 32, 32).contiguous(memory_format=smith.channels_last),),
            # Special encoding of random tensor.
            (smith.utils.bundled_inputs.bundle_randn(1 << 16),),
            # Quantized uniform tensor.
            (smith.quantize_per_tensor(smith.zeros(4, 8, 32, 32), 1, 0, smith.qint8),),
        ]
        smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
            sm, samples, get_expr
        )
        # print(get_expr[0])
        # print(sm._generate_bundled_inputs.code)

        # Make sure the model only grew a little bit,
        # despite having nominally large bundled inputs.
        augmented_size = model_size(sm)

        self.assertLess(augmented_size, original_size + (1 << 12))

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(loaded.get_num_bundled_inputs(), len(samples))
        self.assertEqual(len(inflated), len(samples))

        self.assertTrue(loaded(*inflated[0]) is inflated[0][0])

        for idx, inp in enumerate(inflated):
            self.assertIsInstance(inp, tuple)
            self.assertEqual(len(inp), 1)

            self.assertIsInstance(inp[0], smith.Tensor)
            if idx != 5:
                # Strides might be important for benchmarking.
                self.assertEqual(inp[0].stride(), samples[idx][0].stride())
                self.assertEqual(inp[0], samples[idx][0], exact_dtype=True)

        # This tensor is random, but with 100,000 trials,
        # mean and std had ranges of (-0.0154, 0.0144) and (0.9907, 1.0105).
        self.assertEqual(inflated[5][0].shape, (1 << 16,))
        self.assertEqual(inflated[5][0].mean().item(), 0, atol=0.025, rtol=0)
        self.assertEqual(inflated[5][0].std().item(), 1, atol=0.02, rtol=0)

    def test_large_tensor_with_inflation(self):
        class SingleTensorModel(smith.nn.Module):
            def forward(self, arg):
                return arg

        sm = smith.jit.script(SingleTensorModel())
        sample_tensor = smith.randn(1 << 16)
        # We can store tensors with custom inflation functions regardless
        # of size, even if inflation is just the identity.
        sample = smith.utils.bundled_inputs.bundle_large_tensor(sample_tensor)
        smith.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, [(sample,)])

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(len(inflated), 1)

        self.assertEqual(inflated[0][0], sample_tensor)

    def test_rejected_tensors(self):
        def check_tensor(sample):
            # Need to define the class in this scope to get a fresh type for each run.
            class SingleTensorModel(smith.nn.Module):
                def forward(self, arg):
                    return arg

            sm = smith.jit.script(SingleTensorModel())
            with self.assertRaisesRegex(Exception, "Bundled input argument"):
                smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
                    sm, [(sample,)]
                )

        # Plain old big tensor.
        check_tensor(smith.randn(1 << 16))
        # This tensor has two elements, but they're far apart in memory.
        # We currently cannot represent this compactly while preserving
        # the strides.
        small_sparse = smith.randn(2, 1 << 16)[:, 0:1]
        self.assertEqual(small_sparse.numel(), 2)
        check_tensor(small_sparse)

    def test_non_tensors(self):
        class StringAndIntModel(smith.nn.Module):
            def forward(self, fmt: str, num: int):
                return fmt.format(num)

        sm = smith.jit.script(StringAndIntModel())
        samples = [
            ("first {}", 1),
            ("second {}", 2),
        ]
        smith.utils.bundled_inputs.augment_model_with_bundled_inputs(sm, samples)

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(inflated, samples)

        self.assertTrue(loaded(*inflated[0]) == "first 1")

    def test_multiple_methods_with_inputs(self):
        class MultipleMethodModel(smith.nn.Module):
            def forward(self, arg):
                return arg

            @smith.jit.export
            def foo(self, arg):
                return arg

        mm = smith.jit.script(MultipleMethodModel())
        samples = [
            # Tensor with small numel and small storage.
            (smith.tensor([1]),),
            # Tensor with large numel and small storage.
            (smith.tensor([[2, 3, 4]]).expand(1 << 16, -1)[:, ::2],),
            # Tensor with small numel and large storage.
            (smith.tensor(range(1 << 16))[-8:],),
            # Large zero tensor.
            (smith.zeros(1 << 16),),
            # Large channels-last ones tensor.
            (smith.ones(4, 8, 32, 32).contiguous(memory_format=smith.channels_last),),
        ]
        info = [
            "Tensor with small numel and small storage.",
            "Tensor with large numel and small storage.",
            "Tensor with small numel and large storage.",
            "Large zero tensor.",
            "Large channels-last ones tensor.",
            "Special encoding of random tensor.",
        ]
        smith.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs(
            mm,
            inputs={mm.forward: samples, mm.foo: samples},
            info={mm.forward: info, mm.foo: info},
        )
        loaded = save_and_load(mm)
        inflated = loaded.get_all_bundled_inputs()

        # Make sure these functions are all consistent.
        self.assertEqual(inflated, samples)
        self.assertEqual(inflated, loaded.get_all_bundled_inputs_for_forward())
        self.assertEqual(inflated, loaded.get_all_bundled_inputs_for_foo())

        # Check running and size helpers

        self.assertTrue(loaded(*inflated[0]) is inflated[0][0])
        self.assertEqual(loaded.get_num_bundled_inputs(), len(samples))

        # Check helper that work on all functions
        all_info = loaded.get_bundled_inputs_functions_and_info()
        self.assertEqual(set(all_info.keys()), {"forward", "foo"})
        self.assertEqual(
            all_info["forward"]["get_inputs_function_name"],
            ["get_all_bundled_inputs_for_forward"],
        )
        self.assertEqual(
            all_info["foo"]["get_inputs_function_name"],
            ["get_all_bundled_inputs_for_foo"],
        )
        self.assertEqual(all_info["forward"]["info"], info)
        self.assertEqual(all_info["foo"]["info"], info)

        # example of how to turn the 'get_inputs_function_name' into the actual list of bundled inputs
        for func_name in all_info:
            input_func_name = all_info[func_name]["get_inputs_function_name"][0]
            func_to_run = getattr(loaded, input_func_name)
            self.assertEqual(func_to_run(), samples)

    def test_multiple_methods_with_inputs_both_defined_failure(self):
        class MultipleMethodModel(smith.nn.Module):
            def forward(self, arg):
                return arg

            @smith.jit.export
            def foo(self, arg):
                return arg

        samples = [(smith.tensor([1]),)]

        # inputs defined 2 ways so should fail
        with self.assertRaises(Exception):
            mm = smith.jit.script(MultipleMethodModel())
            definition = textwrap.dedent(
                """
                def _generate_bundled_inputs_for_forward(self):
                    return []
                """
            )
            mm.define(definition)
            smith.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs(
                mm,
                inputs={
                    mm.forward: samples,
                    mm.foo: samples,
                },
            )

    def test_multiple_methods_with_inputs_neither_defined_failure(self):
        class MultipleMethodModel(smith.nn.Module):
            def forward(self, arg):
                return arg

            @smith.jit.export
            def foo(self, arg):
                return arg

        samples = [(smith.tensor([1]),)]

        # inputs not defined so should fail
        with self.assertRaises(Exception):
            mm = smith.jit.script(MultipleMethodModel())
            mm._generate_bundled_inputs_for_forward()
            smith.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs(
                mm,
                inputs={
                    mm.forward: None,
                    mm.foo: samples,
                },
            )

    def test_bad_inputs(self):
        class SingleTensorModel(smith.nn.Module):
            def forward(self, arg):
                return arg

        # Non list for input list
        with self.assertRaises(TypeError):
            m = smith.jit.script(SingleTensorModel())
            smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
                m,
                inputs="foo",  # type: ignore[arg-type]
            )

        # List of non tuples. Most common error using the api.
        with self.assertRaises(TypeError):
            m = smith.jit.script(SingleTensorModel())
            smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
                m,
                inputs=[smith.ones(1, 2)],  # type: ignore[list-item]
            )

    def test_double_augment_fail(self):
        class SingleTensorModel(smith.nn.Module):
            def forward(self, arg):
                return arg

        m = smith.jit.script(SingleTensorModel())
        smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
            m, inputs=[(smith.ones(1),)]
        )
        with self.assertRaisesRegex(
            Exception, "Models can only be augmented with bundled inputs once."
        ):
            smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
                m, inputs=[(smith.ones(1),)]
            )

    def test_double_augment_non_mutator(self):
        class SingleTensorModel(smith.nn.Module):
            def forward(self, arg):
                return arg

        m = smith.jit.script(SingleTensorModel())
        bundled_model = smith.utils.bundled_inputs.bundle_inputs(
            m, inputs=[(smith.ones(1),)]
        )
        with self.assertRaises(AttributeError):
            m.get_all_bundled_inputs()
        self.assertEqual(bundled_model.get_all_bundled_inputs(), [(smith.ones(1),)])
        self.assertEqual(bundled_model.forward(smith.ones(1)), smith.ones(1))

    def test_double_augment_success(self):
        class SingleTensorModel(smith.nn.Module):
            def forward(self, arg):
                return arg

        m = smith.jit.script(SingleTensorModel())
        bundled_model = smith.utils.bundled_inputs.bundle_inputs(
            m, inputs={m.forward: [(smith.ones(1),)]}
        )
        self.assertEqual(bundled_model.get_all_bundled_inputs(), [(smith.ones(1),)])

        bundled_model2 = smith.utils.bundled_inputs.bundle_inputs(
            bundled_model, inputs=[(smith.ones(2),)]
        )
        self.assertEqual(bundled_model2.get_all_bundled_inputs(), [(smith.ones(2),)])

    def test_dict_args(self):
        class MyModel(smith.nn.Module):
            def forward(
                self,
                arg1: Optional[dict[str, smith.Tensor]],
                arg2: Optional[list[smith.Tensor]],
                arg3: smith.Tensor,
            ):
                if arg1 is None:
                    return arg3
                elif arg2 is None:
                    return arg1["a"] + arg1["b"]
                else:
                    return arg1["a"] + arg1["b"] + arg2[0]

        small_sample = dict(
            a=smith.zeros([10, 20]),
            b=smith.zeros([1, 1]),
            c=smith.zeros([10, 20]),
        )
        small_list = [smith.zeros([10, 20])]

        big_sample = dict(
            a=smith.zeros([1 << 5, 1 << 8, 1 << 10]),
            b=smith.zeros([1 << 5, 1 << 8, 1 << 10]),
            c=smith.zeros([1 << 5, 1 << 8, 1 << 10]),
        )
        big_list = [smith.zeros([1 << 5, 1 << 8, 1 << 10])]

        def condensed(t):
            ret = smith.empty_like(t).flatten()[0].clone().expand(t.shape)
            assert ret.storage().size() == 1
            # ret.storage()[0] = 0
            return ret

        def bundle_optional_dict_of_randn(template):
            return smith.utils.bundled_inputs.InflatableArg(
                value=(
                    None
                    if template is None
                    else {k: condensed(v) for (k, v) in template.items()}
                ),
                fmt="{}",
                fmt_fn="""
                def {}(self, value: Optional[Dict[str, Tensor]]):
                    if value is None:
                        return None
                    output = {{}}
                    for k, v in value.items():
                        output[k] = smith.randn_like(v)
                    return output
                """,
            )

        def bundle_optional_list_of_randn(template):
            return smith.utils.bundled_inputs.InflatableArg(
                value=(None if template is None else [condensed(v) for v in template]),
                fmt="{}",
                fmt_fn="""
                def {}(self, value: Optional[List[Tensor]]):
                    if value is None:
                        return None
                    output = []
                    for v in value:
                        output.append(smith.randn_like(v))
                    return output
                """,
            )

        out: list[str] = []
        sm = smith.jit.script(MyModel())
        original_size = model_size(sm)
        small_inputs = (
            bundle_optional_dict_of_randn(small_sample),
            bundle_optional_list_of_randn(small_list),
            smith.zeros([3, 4]),
        )
        big_inputs = (
            bundle_optional_dict_of_randn(big_sample),
            bundle_optional_list_of_randn(big_list),
            smith.zeros([1 << 5, 1 << 8, 1 << 10]),
        )

        smith.utils.bundled_inputs.augment_model_with_bundled_inputs(
            sm,
            [big_inputs, small_inputs],
            _receive_inflate_expr=out,
        )
        augmented_size = model_size(sm)
        # assert the size has not increased more than 8KB

        self.assertLess(augmented_size, original_size + (1 << 13))

        loaded = save_and_load(sm)
        inflated = loaded.get_all_bundled_inputs()
        self.assertEqual(len(inflated[0]), len(small_inputs))

        methods, _ = (
            smith.utils.bundled_inputs._get_bundled_inputs_attributes_and_methods(
                loaded
            )
        )

        # One Function (forward)
        # two bundled inputs (big_inputs and small_inputs)
        # two args which have InflatableArg with fmt_fn
        # 1 * 2 * 2 = 4
        self.assertEqual(
            sum(method.startswith("_inflate_helper") for method in methods), 4
        )


if __name__ == "__main__":
    run_tests()
