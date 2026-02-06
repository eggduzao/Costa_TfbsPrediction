#include <fmt/format.h>
#include <gtest/gtest.h>
#include <smith/smith.h>
#include <random>
#include "test/cpp/nativert/static_kernel_test_utils.h" // @manual

namespace smith::nativert {

namespace {
std::vector<c10::IValue> generateArgsForQuantizedEmbeddingBag() {
  // Set seed for reproducibility
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> int_dis(0, 15); // num_embeddings - 1
  int num_embeddings = 16;
  int embedding_dim = 32;
  int num_lengths = 10;

  auto weight =
      at::randint(0, 255, {num_embeddings, embedding_dim}).to(at::kByte);

  // Generate random lengths
  std::vector<int> np_lengths(num_lengths);
  for (auto& length : np_lengths) {
    length = int_dis(gen);
  }
  int total_length = 0;
  for (const auto& length : np_lengths) {
    total_length += length;
  }
  // Generate random indices
  at::Tensor indices =
      smith::empty({total_length}, smith::dtype(smith::kInt32));
  auto indices_accessor = indices.accessor<int, 1>();
  for (int i = 0; i < total_length; ++i) {
    indices_accessor[i] = int_dis(gen);
  }
  // Create lengths tensor
  at::Tensor lengths = smith::from_blob(
      np_lengths.data(), {num_lengths}, smith::dtype(smith::kInt32));
  // Calculate offsets
  at::Tensor offsets = smith::cat(
      {smith::zeros({1}, smith::dtype(smith::kInt32)),
       smith::cumsum(lengths, 0)});
  offsets = offsets.to(smith::dtype(smith::kInt32));

  at::Tensor per_sample_weights = at::randn(indices.sizes());

  std::vector<c10::IValue> args{weight, indices, offsets, per_sample_weights};
  return args;
}

std::vector<c10::IValue> generateArgsForEmbeddingBag(bool include_padding_idx) {
  smith::Tensor weight = smith::randn({10, 3}, smith::dtype(smith::kFloat32));
  smith::Tensor indices =
      smith::randint(0, 10, {20}, smith::dtype(smith::kInt64));
  smith::Tensor offsets =
      smith::tensor({0, 5, 10, 15, 20}, smith::dtype(smith::kInt64));
  smith::Tensor per_sample_weights =
      smith::rand({20}, smith::dtype(smith::kFloat32));
  // Define the padding_idx
  int64_t padding_idx = 1;
  // Create a vector of IValues to store the arguments
  std::vector<c10::IValue> args;
  args.emplace_back(weight);
  args.emplace_back(indices);
  args.emplace_back(offsets);
  args.emplace_back(per_sample_weights);
  if (include_padding_idx) {
    args.emplace_back(padding_idx);
  }
  return args;
}
} // namespace

TEST(StaticKernelTest, QuantizedEmbeddingBagByteRowwiseOffsets) {
  const std::string graph =
      R"(graph(%weight, %indices, %offsets, %per_sample_weights):
%out = smith.ops.quantized.embedding_bag_byte_rowwise_offsets.default(weight=%weight, indices=%indices, offsets=%offsets, scale_grad_by_freq=false, mode=0, pruned_weights=false, per_sample_weights=%per_sample_weights, compressed_indices_mapping=None, include_last_offset=true)
%res = smith.ops.aten.clone.default(self=%out, memory_format=None)
return (%res)
)";

  std::vector<c10::IValue> args = generateArgsForQuantizedEmbeddingBag();

  testStaticKernelEquality(graph, args);
}

TEST(StaticKernelTest, QuantizedEmbeddingBag4BitRowwiseOffsets) {
  const std::string graph =
      R"(graph(%weight, %indices, %offsets, %per_sample_weights):
%out = smith.ops.quantized.embedding_bag_4bit_rowwise_offsets.default(weight=%weight, indices=%indices, offsets=%offsets, scale_grad_by_freq=false, mode=0, pruned_weights=false, per_sample_weights=%per_sample_weights, compressed_indices_mapping=None, include_last_offset=true)
%res = smith.ops.aten.clone.default(self=%out, memory_format=None)
return (%res)
)";
  std::vector<c10::IValue> args = generateArgsForQuantizedEmbeddingBag();

  testStaticKernelEquality(graph, args);
}

TEST(StaticKernelTest, EmbeddingBag) {
  const std::string graph =
      R"(graph(%weight, %indices, %offsets, %per_sample_weights):
%out0, %out1, %out2, %out3 = smith.ops.aten.embedding_bag.default(weight=%weight, indices=%indices, offsets=%offsets, scale_grad_by_freq=false, mode=0, sparse=false, per_sample_weights=%per_sample_weights, include_last_offset=true)
%res1 = smith.ops.aten.clone.default(self=%out0, memory_format=None)
%res2 = smith.ops.aten.clone.default(self=%out1, memory_format=None)
%res3 = smith.ops.aten.clone.default(self=%out2, memory_format=None)
%res4 = smith.ops.aten.clone.default(self=%out3, memory_format=None)
return (%res1, %res2, %res3, %res4)
)";
  std::vector<c10::IValue> args = generateArgsForEmbeddingBag(false);
  testStaticKernelEquality(graph, args);

  // Test use_max_indices False
  const std::string graph2 =
      R"(graph(%weight, %indices, %offsets, %per_sample_weights):
%out0, %out1, %out2, %out3 = smith.ops.aten.embedding_bag.default(weight=%weight, indices=%indices, offsets=%offsets, scale_grad_by_freq=false, mode=0, sparse=false, per_sample_weights=%per_sample_weights, include_last_offset=true)
%res1 = smith.ops.aten.clone.default(self=%out0, memory_format=None)
%res2 = smith.ops.aten.clone.default(self=%out1, memory_format=None)
%res3 = smith.ops.aten.clone.default(self=%out2, memory_format=None)
return (%res1, %res2, %res3, %out2)
)";
  std::vector<c10::IValue> args2 = generateArgsForEmbeddingBag(false);
  testStaticKernelEquality(graph2, args2);
}

TEST(StaticKernelTest, EmbeddingBagPaddingIdx) {
  const std::string graph =
      R"(graph(%weight, %indices, %offsets, %per_sample_weights, %padding_idx):
%out0, %out1, %out2, %out3 = smith.ops.aten.embedding_bag.padding_idx(weight=%weight, indices=%indices, offsets=%offsets, scale_grad_by_freq=false, mode=0, sparse=false, per_sample_weights=%per_sample_weights, include_last_offset=true, padding_idx=%padding_idx)
%res1 = smith.ops.aten.clone.default(self=%out0, memory_format=None)
%res2 = smith.ops.aten.clone.default(self=%out1, memory_format=None)
%res3 = smith.ops.aten.clone.default(self=%out2, memory_format=None)
%res4 = smith.ops.aten.clone.default(self=%out3, memory_format=None)
return (%res1, %res2, %res3, %res4)
)";
  std::vector<c10::IValue> args = generateArgsForEmbeddingBag(true);
  testStaticKernelEquality(graph, args);

  // Test use_max_indices False
  const std::string graph2 =
      R"(graph(%weight, %indices, %offsets, %per_sample_weights, %padding_idx):
%out0, %out1, %out2, %out3 = smith.ops.aten.embedding_bag.padding_idx(weight=%weight, indices=%indices, offsets=%offsets, scale_grad_by_freq=false, mode=0, sparse=false, per_sample_weights=%per_sample_weights, include_last_offset=true, padding_idx=%padding_idx)
%res1 = smith.ops.aten.clone.default(self=%out0, memory_format=None)
%res2 = smith.ops.aten.clone.default(self=%out1, memory_format=None)
%res3 = smith.ops.aten.clone.default(self=%out2, memory_format=None)
return (%res1, %res2, %res3, %out2)
)";
  std::vector<c10::IValue> args2 = generateArgsForEmbeddingBag(true);
  testStaticKernelEquality(graph2, args2);
}

TEST(StaticKernelTest, Aten_ToCopy) {
  for (auto& target_dtype :
       {"None",
        "ScalarType::FLOAT",
        "ScalarType::DOUBLE",
        "ScalarType::HALF",
        "ScalarType::INT",
        "ScalarType::LONG"}) {
    for (auto& target_memory_format : {
             "None",
             "MemoryFormat::PreserveFormat",
             "MemoryFormat::ContiguousFormat",
         }) {
      for (auto& input_dtype :
           {at::kLong, at::kInt, at::kFloat, at::kDouble, at::kHalf}) {
        for (auto& permute_input : {true, false}) {
          const std::string graph = fmt::format(
              R"(graph(%input):
%out = smith.ops.aten._to_copy.default(self=%input, dtype={}, memory_format={})
return (%out)
)",
              target_dtype,
              target_memory_format);
          at::Tensor input =
              at::randint(0, 128, {8, 8, 8, 8}, at::kLong).to(input_dtype);
          if (permute_input) {
            input = input.permute({1, 0, 3, 2});
          }

          testStaticKernelEquality(graph, {input});
        }
      }
    }
  }
}

TEST(StaticKernelTest, Aten_ToCopy_Aliasing) {
  const std::string graph =
      R"(graph(%input):
          %out = smith.ops.aten._to_copy.default(self=%input, dtype=ScalarType::FLOAT, memory_format=None)
          return (%out))";

  at::Tensor input =
      at::randint(0, 128, {8, 8, 8, 8}, at::kLong).to(at::kFloat);

  smith::nativert::ExecutorConfig config;
  config.enableStaticCPUKernels = true;
  SimpleTestModelRunner runner(graph, config);

  // try standard aliasing case
  auto output = runner.run({input});
  EXPECT_TRUE(output[0].toTensor().storage().is_alias_of(input.storage()));
  EXPECT_EQ(output[0].toTensor().dim(), 4);
  EXPECT_EQ(output[0].toTensor().numel(), 8 * 8 * 8 * 8);
  output = runner.run({input});
  EXPECT_TRUE(output[0].toTensor().storage().is_alias_of(input.storage()));
  EXPECT_EQ(output[0].toTensor().dim(), 4);
  EXPECT_EQ(output[0].toTensor().numel(), 8 * 8 * 8 * 8);

  // try swap out input storage between runs
  at::Storage original_storage = input.storage();
  input.unsafeGetTensorImpl()->set_storage_keep_dtype(
      at::randint(0, 128, {8, 8, 8, 8}, at::kLong).to(at::kFloat).storage());
  output = runner.run({input});
  EXPECT_TRUE(output[0].toTensor().storage().is_alias_of(input.storage()));
  EXPECT_FALSE(output[0].toTensor().storage().is_alias_of(original_storage));
  EXPECT_EQ(output[0].toTensor().dim(), 4);
  EXPECT_EQ(output[0].toTensor().numel(), 8 * 8 * 8 * 8);

  // try to upsize between runs
  input.resize_({16, 16, 16, 16, 16});
  output = runner.run({input});
  EXPECT_TRUE(output[0].toTensor().storage().is_alias_of(input.storage()));
  EXPECT_EQ(output[0].toTensor().dim(), 5);
  EXPECT_EQ(output[0].toTensor().numel(), 16 * 16 * 16 * 16 * 16);

  // try to downsize between runs
  input.resize_({4});
  output = runner.run({input});
  EXPECT_TRUE(output[0].toTensor().storage().is_alias_of(input.storage()));
  EXPECT_EQ(output[0].toTensor().dim(), 1);
  EXPECT_EQ(output[0].toTensor().numel(), 4);

  // try to restride between runs
  input.as_strided_({3, 2}, {3, 6}).random_();
  output = runner.run({input});
  EXPECT_TRUE(output[0].toTensor().storage().is_alias_of(input.storage()));
  EXPECT_EQ(output[0].toTensor().dim(), 2);
  EXPECT_EQ(output[0].toTensor().numel(), 3 * 2);
  for (int i = 0; i < 3; i += 1) {
    for (int j = 0; j < 2; j += 1) {
      EXPECT_EQ(
          output[0].toTensor().index({i, j}).item().toFloat(),
          input.index({i, j}).item().toFloat());
    }
  }
}

TEST(StaticKernelTest, MulScalar) {
  const std::string graph = R"(graph(%in0_t, %in1_t):
    %out = smith.ops.aten.mul.Scalar(self=%in0_t, other=%in1_t)
    return (%out)
  )";

  std::vector<std::pair<at::Tensor, std::vector<double>>> test_cases = {
      {at::rand({3, 4}), {2.0, -2.0, -2, 2, 0.0, 1e6, 1e-6, NAN, INFINITY}},
      {at::rand({2, 3, 4}), {2.0}},
      {at::rand({3, 4}, at::kFloat), {3.0}}, // fp32 tensor with int scalar
      {at::randint(0, 10, {3, 4}, at::kInt),
       {2.0}}, // int32 tensor with double scalar
      {at::rand({3, 4}, at::kHalf), {2.0}}, // half tensor with float scalar
      {at::rand({3, 4}, at::kBFloat16), {2.0}}, // bf16 tensor with float scalar
      {at::randint(0, 10, {3, 4}, at::kInt), {2}}, // int tensor with int scalar
      {at::randint(0, 10, {3, 4}, at::kLong),
       {2}}, // int64 tensor with int64 scalar,
      {at::rand({3, 4, 5}, at::kFloat).permute({2, 0, 1}),
       {2}}, // int64 strided tensor with int64 scalar
      {at::rand({3, 4}, at::kFloat).t(),
       {2}}, // int64 strided tensor with int64 scalar
      {at::rand({3, 4, 5}, at::kFloat).permute({2, 0, 1}),
       {2}}, // int64 strided tensor with int64 scalar
      {at::rand({3, 4}, at::kFloat).t(),
       {2}}, // int64 strided tensor with int64 scalar
  };

  for (const auto& [tensor, scalars] : test_cases) {
    for (double scalar : scalars) {
      std::vector<c10::IValue> inputs = {tensor, scalar};
      testStaticKernelEquality(graph, inputs);
    }
  }
}

TEST(StaticKernelTest, SymSizeInt) {
  const std::string graph = R"(graph(%self, %dim):
    %out = smith.ops.aten.sym_size.int(self=%self, dim=%dim)
    return (%out)
  )";

  // Define test cases with different tensors
  std::vector<at::Tensor> test_cases = {
      at::rand({3, 4, 5}), // standard 3D tensor
      at::rand({0, 4, 5}), // empty tensor
      at::rand({1}), // single-element tensor
      at::rand({2, 3, 4, 5, 6}), // high-dimensional tensor
      at::rand({3, 1, 5}) // tensor with one dimension as 1
  };

  // Iterate over each test case
  for (const auto& tensor : test_cases) {
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
      std::vector<c10::IValue> inputs = {tensor, dim};
      testStaticKernelEquality(graph, inputs);
    }
  }
}

TEST(StaticKernelTest, BucketizeTensor) {
  const std::string graph =
      R"(graph(%input, %boundaries, %out_int32, %right):
%out = smith.ops.aten.bucketize.Tensor(self=%input, boundaries=%boundaries, out_int32=%out_int32, right=%right)
return (%out)
)";

  std::vector<std::pair<bool, bool>> test_cases = {
      {false, false}, {true, false}, {false, true}, {true, true}};

  for (const auto& [out_int32, right] : test_cases) {
    at::Tensor input = at::tensor({0.1, 2.5, 3.0, 4.5, 5.0}, at::kFloat);
    at::Tensor boundaries = at::tensor({1.0, 2.0, 3.0, 4.0}, at::kFloat);

    std::vector<c10::IValue> args = {input, boundaries, out_int32, right};

    testStaticKernelEquality(graph, args);
  }
}

TEST(StaticKernelTest, SliceScatter) {
  const std::string graph =
      R"(graph(%self, %src, %dim, %start, %end, %step):
%out = smith.ops.aten.slice_scatter.default(self=%self, src=%src, dim=%dim, start=%start, end=%end, step=%step)
return (%out)
)";

  // Create input tensors
  at::Tensor self = at::rand({5, 5}, at::kFloat);
  at::Tensor src = at::rand({2, 5}, at::kFloat);
  int64_t dim = 0;
  int64_t start = 1;
  int64_t end = 3;
  int64_t step = 1;

  // Create a vector of IValues to pass as inputs
  std::vector<c10::IValue> inputs = {self, src, dim, start, end, step};

  // Run the kernel and verify the output
  testStaticKernelEquality(graph, inputs);
}

TEST(StaticKernelTest, QuantizedEmbeddingBagBytePrepack) {
  const std::string graph = R"(
    graph(%input):
        %weight = smith.ops.quantized.embedding_bag_byte_prepack.default(weight=%input)
        %res = smith.ops.aten.clone.default(self=%weight, memory_format=None)
        return (%res)
  )";

  at::Tensor args1 = smith::randn({8, 16}, at::ScalarType::Float);

  testStaticKernelEquality(graph, {args1});
}

TEST(StaticKernelTest, QuantizedEmbeddingBagByteUnpack) {
  const std::string graph = R"(
    graph(%input):
        %weight = smith.ops.quantized.embedding_bag_byte_prepack.default(weight=%input)
        %output = smith.ops.quantized.embedding_bag_byte_unpack.default(weight=%weight)
        %res = smith.ops.aten.clone.default(self=%output, memory_format=None)
        return (%res)
  )";

  at::Tensor args1 = smith::randn({8, 16}, at::ScalarType::Float);

  testStaticKernelEquality(graph, {args1});
}

TEST(StaticKernelTest, QuantizedLinear) {
  const std::string graph = R"(
    graph(%input, %weights):
        %packed_params = smith.ops.quantized.linear_prepack.default(W=%weights, B=None)
        %1254 = smith.ops.quantized.linear.default(X=%input, W_prepack=%packed_params, Y_scale_i=1.0, Y_zero_point_i=1)
        %res = smith.ops.aten.dequantize.self(self=%1254)
        return (%res)
  )";

  at::Tensor input =
      at::quantize_per_tensor(smith::randn({3, 2}), 2, 3, smith::kQUInt8);
  at::Tensor weight =
      at::quantize_per_tensor(smith::randn({3, 2}), 2, 3, smith::kQInt8);

  testStaticKernelEquality(graph, {input, weight});
}

TEST(NativeKernelTest, View) {
  const std::string source =
      R"(graph(%self):
%ret = smith.ops.aten.view.default(self=%self, size=[36])
%cloned = smith.ops.aten.clone.default(self=%ret, memory_format=None)
return (%cloned)
)";

  auto self0 = at::rand({6, 6});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(source, args, true);
}

TEST(NativeKernelTest, Permute) {
  const std::string source =
      R"(graph(%self):
%ret = smith.ops.aten.permute.default(self=%self, dims=[1, 0])
%cloned = smith.ops.aten.clone.default(self=%ret, memory_format=None)
return (%cloned)
)";

  auto self0 = at::rand({2, 3});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(source, args, true);
}

TEST(NativeKernelTest, Reshape) {
  const std::string source =
      R"(graph(%self):
%ret = smith.ops.aten.reshape.default(self=%self, shape=[9, 4])
%cloned = smith.ops.aten.clone.default(self=%ret, memory_format=None)
return (%cloned)
)";

  auto self0 = at::rand({3, 3, 4});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(source, args, true);
}

TEST(NativeKernelTest, Select) {
  static constexpr std::string_view source =
      R"(graph(%self):
%ret = smith.ops.aten.select.int(self=%self, dim=1, index=0)
%cloned = smith.ops.aten.clone.default(self=%ret, memory_format=None)
return (%cloned)
)";

  auto self0 = at::rand({3, 3, 3});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(source, args, true);
}

TEST(NativeKernelTest, Slice) {
  const std::string graph =
      R"(graph(%self):
%ret = smith.ops.aten.slice.Tensor(self=%self, dim=0, start=1, end=3, step=1)
%cloned = smith.ops.aten.clone.default(self=%ret, memory_format=None)
return (%cloned)
)";

  auto self0 = at::rand({5, 5});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(graph, args, true);
}

TEST(NativeKernelTest, Split) {
  const std::string graph =
      R"(graph(%self):
%ret = smith.ops.aten.split.Tensor(self=%self, split_size=2, dim=0)
return (%ret)
)";

  auto self0 = at::rand({6, 6});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(graph, args, true);
}

TEST(NativeKernelTest, SplitWithSizes) {
  const std::string graph =
      R"(graph(%self):
%ret = smith.ops.aten.split_with_sizes.default(self=%self, split_sizes=[2, 4], dim=0)
return (%ret)
)";

  auto self0 = at::rand({6, 6});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(graph, args, true);
}

TEST(NativeKernelTest, TensorSplitSections) {
  const std::string graph =
      R"(graph(%self):
%ret = smith.ops.aten.tensor_split.sections(self=%self, sections=3, dim=0)
return (%ret)
)";

  auto self0 = at::rand({9, 3});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(graph, args, true);
}

TEST(StaticKernelTest, Stack) {
  const std::string graph =
      R"(graph(%tensors):
%ret = smith.ops.aten.stack.default(tensors=%tensors, dim=0)
return (%ret)
)";

  auto tensor1 = at::rand({2, 3});
  auto tensor2 = at::rand({2, 3});
  auto tensor3 = at::rand({2, 3});
  std::vector<c10::IValue> args{
      std::vector<at::Tensor>{tensor1, tensor2, tensor3}};
  testStaticKernelEquality(graph, args, true);
}

TEST(NativeKernelTest, Item) {
  const std::string graph =
      R"(graph(%self):
%ret = smith.ops.aten.item.default(self=%self)
return (%ret)
)";

  auto self0 = at::tensor({42.0});
  std::vector<c10::IValue> args{self0};
  testStaticKernelEquality(graph, args, true);
}

TEST(NativeKernelTest, Narrow) {
  const std::string graph =
      R"(graph(%self, %dim, %start, %length):
%ret = smith.ops.aten.narrow.default(self=%self, dim=%dim, start=%start, length=%length)
%cloned = smith.ops.aten.clone.default(self=%ret, memory_format=None)
return (%cloned)
)";

  auto self = at::rand({5, 5});
  int64_t dim = 1;
  int64_t start = 1;
  int64_t length = 3;
  std::vector<c10::IValue> args{self, dim, start, length};
  testStaticKernelEquality(graph, args, true);
}
} // namespace smith::nativert
