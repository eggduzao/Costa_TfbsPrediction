#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <smith/smith.h>

#include <test/cpp/api/support.h>

#include <smith/expanding_array.h>
#include <smith/nn/functional/activation.h>
#include <smith/nn/options/activation.h>
#include <limits>
#include <random>

using namespace smith::nn;
using namespace smith::test;

class TestModel : public smith::nn::Module {
 public:
  TestModel()
      : l1(register_module("l1", Linear(10, 3))),
        l2(register_module("l2", Linear(3, 5))),
        l3(register_module("l3", Linear(5, 100))) {}

  Linear l1, l2, l3;
};

class NestedModel : public smith::nn::Module {
 public:
  NestedModel()
      : param_(register_parameter("param", smith::empty({3, 2, 21}))),
        l1(register_module("l1", Linear(5, 20))),
        t(register_module("test", std::make_shared<TestModel>())) {}

  smith::Tensor param_;
  Linear l1;
  std::shared_ptr<TestModel> t;
};

struct ModulesTest : smith::test::SeedingFixture {};

TEST_F(ModulesTest, Conv1d) {
  Conv1d model(Conv1dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(
      smith::arange(18, smith::dtype(smith::kFloat)).reshape({2, 3, 3}));
  auto x = smith::arange(30, smith::dtype(smith::kFloat).requires_grad(true))
               .reshape({2, 3, 5});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{312., 348., 384.}, {798., 915., 1032.}},

       {{852., 888., 924.}, {2553., 2670., 2787.}}},
      smith::kFloat);
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3);
}

TEST_F(ModulesTest, Conv1dSameStrided) {
  auto options = Conv1dOptions(3, 2, 3);
  options.stride(1).padding(smith::kSame);
  Conv1d model_valid(options);
  ASSERT_THROWS_WITH(
      [&] { Conv1d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, Conv1dIvalidArg) {
  auto options = Conv1dOptions(3, 2, 3).groups(-1);
  ASSERT_THROWS_WITH(
      Conv1d(options), "in_channels, groups and out_channels must");
}

TEST_F(ModulesTest, Conv2dEven) {
  Conv2d model(Conv2dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(
      smith::arange(54, smith::dtype(smith::kFloat)).reshape({2, 3, 3, 3}));
  auto x = smith::arange(75, smith::dtype(smith::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{{15219., 15570., 15921.},
         {16974., 17325., 17676.},
         {18729., 19080., 19431.}},

        {{37818., 38898., 39978.},
         {43218., 44298., 45378.},
         {48618., 49698., 50778.}}}},
      smith::kFloat);
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 3);
}

TEST_F(ModulesTest, Conv2dUneven) {
  Conv2d model(Conv2dOptions(3, 2, {3, 2}).stride({1, 1}).bias(false));
  model->weight.set_data(
      smith::arange(36, smith::dtype(smith::kFloat)).reshape({2, 3, 3, 2}));
  auto x = smith::arange(60, smith::dtype(smith::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 4});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{{5289., 5442., 5595.}, {5901., 6054., 6207.}, {6513., 6666., 6819.}},

        {{13227., 13704., 14181.},
         {15135., 15612., 16089.},
         {17043., 17520., 17997.}}}},
      smith::kFloat);
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, Conv2dSameStrided) {
  auto options = Conv2dOptions(3, 2, {3, 4});
  options.stride(1).padding(smith::kSame);
  Conv2d model_valid(options);
  ASSERT_THROWS_WITH(
      [&] { Conv2d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
  ASSERT_THROWS_WITH(
      [&] { Conv2d model_invalid(options.stride({1, 2})); }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, Conv3d) {
  Conv3d model(Conv3dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(
      smith::arange(162, smith::dtype(smith::kFloat)).reshape({2, 3, 3, 3, 3}));
  auto x = smith::arange(375, smith::dtype(smith::kFloat).requires_grad(true))
               .reshape({1, 3, 5, 5, 5});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{{{700704., 703944., 707184.},
          {716904., 720144., 723384.},
          {733104., 736344., 739584.}},

         {{781704., 784944., 788184.},
          {797904., 801144., 804384.},
          {814104., 817344., 820584.}},

         {{862704., 865944., 869184.},
          {878904., 882144., 885384.},
          {895104., 898344., 901584.}}},

        {{{1724220., 1734021., 1743822.},
          {1773225., 1783026., 1792827.},
          {1822230., 1832031., 1841832.}},

         {{1969245., 1979046., 1988847.},
          {2018250., 2028051., 2037852.},
          {2067255., 2077056., 2086857.}},

         {{2214270., 2224071., 2233872.},
          {2263275., 2273076., 2282877.},
          {2312280., 2322081., 2331882.}}}}},
      smith::kFloat);
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_TRUE(model->weight.grad().numel() == 3 * 2 * 3 * 3 * 3);
}

TEST_F(ModulesTest, Conv3dSameStrided) {
  auto options = Conv3dOptions(3, 2, {3, 4, 5});
  options.stride(1).padding(smith::kSame);
  Conv3d model_valid(options);
  ASSERT_THROWS_WITH(
      [&] { Conv3d model_invalid(options.stride(2)); }(),
      "padding='same' is not supported for strided convolutions");
  ASSERT_THROWS_WITH(
      [&] { Conv3d model_invalid(options.stride({1, 2, 1})); }(),
      "padding='same' is not supported for strided convolutions");
}

TEST_F(ModulesTest, ConvTranspose1d) {
  ConvTranspose1d model(ConvTranspose1dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(smith::arange(18.).view({2, 3, 3}));
  auto x = smith::arange(20.).reshape({2, 2, 5});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{45., 104., 179., 212., 245., 188., 107.},
        {60., 140., 242., 293., 344., 260., 146.},
        {75., 176., 305., 374., 443., 332., 185.}},
       {{135., 304., 509., 542., 575., 428., 237.},
        {210., 460., 752., 803., 854., 620., 336.},
        {285., 616., 995., 1064., 1133., 812., 435.}}});
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3);
}

TEST_F(ModulesTest, ConvTranspose2dEven) {
  ConvTranspose2d model(ConvTranspose2dOptions(3, 2, 3).stride(1).bias(false));
  model->weight.set_data(smith::arange(54.).view({2, 3, 3, 3}));
  auto x = smith::arange(50.).view({1, 2, 5, 5});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{{675., 1402., 2183., 2270., 2357., 1634., 849.},
         {1560., 3240., 5044., 5236., 5428., 3760., 1952.},
         {2685., 5574., 8673., 8988., 9303., 6438., 3339.},
         {3180., 6594., 10248., 10563., 10878., 7518., 3894.},
         {3675., 7614., 11823., 12138., 12453., 8598., 4449.},
         {2820., 5832., 9040., 9268., 9496., 6544., 3380.},
         {1605., 3314., 5129., 5252., 5375., 3698., 1907.}},
        {{900., 1870., 2912., 3053., 3194., 2210., 1146.},
         {2100., 4356., 6772., 7072., 7372., 5092., 2636.},
         {3630., 7518., 11670., 12147., 12624., 8706., 4500.},
         {4395., 9078., 14055., 14532., 15009., 10326., 5325.},
         {5160., 10638., 16440., 16917., 17394., 11946., 6150.},
         {3900., 8028., 12388., 12724., 13060., 8956., 4604.},
         {2190., 4502., 6938., 7115., 7292., 4994., 2564.}},
        {{1125., 2338., 3641., 3836., 4031., 2786., 1443.},
         {2640., 5472., 8500., 8908., 9316., 6424., 3320.},
         {4575., 9462., 14667., 15306., 15945., 10974., 5661.},
         {5610., 11562., 17862., 18501., 19140., 13134., 6756.},
         {6645., 13662., 21057., 21696., 22335., 15294., 7851.},
         {4980., 10224., 15736., 16180., 16624., 11368., 5828.},
         {2775., 5690., 8747., 8978., 9209., 6290., 3221.}}}});
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 3);
}

TEST_F(ModulesTest, ConvTranspose2dUneven) {
  ConvTranspose2d model(
      ConvTranspose2dOptions(3, 2, {3, 2}).stride({1, 1}).bias(false));
  model->weight.set_data(smith::arange(36.).view({2, 3, 3, 2}));
  auto x = smith::arange(40.).view({1, 2, 5, 4});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{{360., 758., 796., 834., 440.},
         {832., 1752., 1836., 1920., 1012.},
         {1432., 3014., 3152., 3290., 1732.},
         {1696., 3566., 3704., 3842., 2020.},
         {1960., 4118., 4256., 4394., 2308.},
         {1504., 3152., 3252., 3352., 1756.},
         {856., 1790., 1844., 1898., 992.}},
        {{480., 1010., 1072., 1134., 596.},
         {1120., 2352., 2484., 2616., 1372.},
         {1936., 4058., 4268., 4478., 2344.},
         {2344., 4898., 5108., 5318., 2776.},
         {2752., 5738., 5948., 6158., 3208.},
         {2080., 4328., 4476., 4624., 2404.},
         {1168., 2426., 2504., 2582., 1340.}},
        {{600., 1262., 1348., 1434., 752.},
         {1408., 2952., 3132., 3312., 1732.},
         {2440., 5102., 5384., 5666., 2956.},
         {2992., 6230., 6512., 6794., 3532.},
         {3544., 7358., 7640., 7922., 4108.},
         {2656., 5504., 5700., 5896., 3052.},
         {1480., 3062., 3164., 3266., 1688.}}}});
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(model->weight.grad().numel(), 3 * 2 * 3 * 2);
}

TEST_F(ModulesTest, ConvTranspose3d) {
  ConvTranspose3d model(ConvTranspose3dOptions(2, 2, 2).stride(1).bias(false));
  model->weight.set_data(smith::arange(32.).reshape({2, 2, 2, 2, 2}));
  auto x = smith::arange(16.).reshape({1, 2, 2, 2, 2});
  auto y = model(x);
  auto expected = smith::tensor(
      {{{{{128., 280., 154.}, {304., 664., 364.}, {184., 400., 218.}},
         {{352., 768., 420.}, {832., 1808., 984.}, {496., 1072., 580.}},
         {{256., 552., 298.}, {592., 1272., 684.}, {344., 736., 394.}}},
        {{{192., 424., 234.}, {464., 1016., 556.}, {280., 608., 330.}},
         {{544., 1184., 644.}, {1280., 2768., 1496.}, {752., 1616., 868.}},
         {{384., 824., 442.}, {880., 1880., 1004.}, {504., 1072., 570.}}}}});
  ASSERT_TRUE(smith::allclose(y, expected));

  smith::Tensor s = y.sum();
  s.backward();
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_TRUE(model->weight.grad().numel() == 2 * 2 * 2 * 2 * 2);
}

TEST_F(ModulesTest, MaxPool1d) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = smith::ones({1, 1, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({1, 1, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool1dReturnIndices) {
  MaxPool1d model(MaxPool1dOptions(3).stride(2));
  auto x = smith::ones({1, 1, 5}, smith::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({1, 1, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));

  ASSERT_TRUE(
      smith::allclose(indices, smith::tensor({{{0, 2}}}, smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, MaxPool2dEven) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = smith::ones({2, 5, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dUneven) {
  MaxPool2d model(MaxPool2dOptions({3, 2}).stride({2, 2}));
  auto x = smith::ones({2, 5, 4}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool2dReturnIndices) {
  MaxPool2d model(MaxPool2dOptions(3).stride(2));
  auto x = smith::ones({2, 5, 5}, smith::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
  ASSERT_TRUE(smith::allclose(
      indices,
      smith::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}}, smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3d) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = smith::ones({2, 5, 5, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, MaxPool3dReturnIndices) {
  MaxPool3d model(MaxPool3dOptions(3).stride(2));
  auto x = smith::ones({2, 5, 5, 5}, smith::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  ASSERT_TRUE(smith::allclose(
      indices,
      smith::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}}},
          smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool1d) {
  AvgPool1d model(AvgPool1dOptions(3).stride(2));
  auto x = smith::ones({1, 1, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({1, 1, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 2}));
}

TEST_F(ModulesTest, AvgPool2dEven) {
  AvgPool2d model(AvgPool2dOptions(3).stride(2));
  auto x = smith::ones({2, 5, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool2dUneven) {
  AvgPool2d model(AvgPool2dOptions({3, 2}).stride({2, 2}));
  auto x = smith::ones({2, 5, 4}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, AvgPool3d) {
  AvgPool3d model(AvgPool3dOptions(3).stride(2));
  auto x = smith::ones({2, 5, 5, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool2d) {
  FractionalMaxPool2d model(FractionalMaxPool2dOptions(3).output_size(2));
  auto x = smith::ones({2, 5, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool2dReturnIndices) {
  FractionalMaxPool2d model(FractionalMaxPool2dOptions(3).output_size(2));
  auto x = smith::ones({2, 5, 5}, smith::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2}));
  ASSERT_TRUE(smith::allclose(
      indices, smith::tensor({{{0, 2}, {10, 12}}, {{0, 2}, {10, 12}}})));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool3d) {
  FractionalMaxPool3d model(FractionalMaxPool3dOptions(3).output_size(2));
  auto x = smith::ones({2, 5, 5, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2, 2})));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, FractionalMaxPool3dReturnIndices) {
  FractionalMaxPool3d model(FractionalMaxPool3dOptions(3).output_size(2));
  auto x = smith::ones({2, 5, 5, 5}, smith::requires_grad());
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(smith::allclose(y, smith::ones({2, 2, 2, 2})));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 2, 2, 2}));

  ASSERT_TRUE(smith::allclose(
      indices,
      smith::tensor(
          {{{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}},
           {{{0, 2}, {10, 12}}, {{50, 52}, {60, 62}}}})));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 2, 2, 2}));
}

TEST_F(ModulesTest, LPPool1d) {
  int norm_type = 2;
  int stride = 2;
  int kernel_size = 3;

  LPPool1d model(LPPool1dOptions(norm_type, kernel_size).stride(stride));
  auto x = smith::ones({1, 1, 5});
  auto y = model(x);
  auto expected =
      (smith::pow(smith::tensor({{{1, 1}}}, smith::kFloat), norm_type) *
       kernel_size)
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, expected));
  ASSERT_EQ(y.sizes(), smith::IntArrayRef({1, 1, 2}));
}

TEST_F(ModulesTest, LPPool2d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({2, 3});

  LPPool2d model(LPPool2dOptions(norm_type, kernel_size).stride(stride));
  auto x = smith::ones({1, 1, 2, 5});
  auto y = model(x);
  auto expected =
      (smith::pow(smith::tensor({{{{1, 1}}}}, smith::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1]))
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(y, expected));
  ASSERT_EQ(y.sizes(), smith::IntArrayRef({1, 1, 1, 2}));
}

TEST_F(ModulesTest, LPPool3d) {
  int norm_type = 2;
  int stride = 2;
  std::vector<int64_t> kernel_size({1, 2, 3});

  LPPool3d model(LPPool3dOptions(norm_type, kernel_size).stride(stride));
  auto x = smith::ones({1, 1, 1, 2, 5});
  auto y = model(x);
  auto expected =
      (smith::pow(smith::tensor({{{{{1, 1}}}}}, smith::kFloat), norm_type) *
       (kernel_size[0] * kernel_size[1] * kernel_size[2]))
          .pow(1. / norm_type);

  ASSERT_EQ(y.ndimension(), 5);
  ASSERT_TRUE(smith::allclose(y, expected));
  ASSERT_EQ(y.sizes(), smith::IntArrayRef({1, 1, 1, 1, 2}));
}

TEST_F(ModulesTest, Identity) {
  Identity identity;
  auto input = smith::tensor(
      {{1, 3, 4}, {2, 3, 4}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto output = identity->forward(input);
  auto expected = smith::tensor({{1, 3, 4}, {2, 3, 4}}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(smith::equal(output, expected));
  ASSERT_TRUE(smith::equal(input.grad(), smith::ones_like(input)));
}

TEST_F(ModulesTest, Flatten) {
  Flatten flatten;
  auto input = smith::tensor(
      {{1, 3, 4}, {2, 5, 6}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto output = flatten->forward(input);
  auto expected = smith::tensor({{1, 3, 4}, {2, 5, 6}}, smith::kFloat);
  auto s = output.sum();

  s.backward();
  ASSERT_TRUE(smith::equal(output, expected));
  ASSERT_TRUE(smith::equal(input.grad(), smith::ones_like(input)));

  // Testing with optional arguments start_dim and end_dim
  Flatten flatten_optional_dims(FlattenOptions().start_dim(2).end_dim(3));
  input = smith::tensor(
      {{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
       {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}},
      smith::dtype(smith::kFloat)
          .requires_grad(true)); // Tensor with sizes (2, 2, 2, 2)

  output = flatten_optional_dims->forward(input);
  expected = smith::tensor(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}}, {{9, 10, 11, 12}, {13, 14, 15, 16}}},
      smith::kFloat); // Tensor with sizes (2, 2, 4)

  s = output.sum();
  s.backward();
  ASSERT_TRUE(smith::equal(output, expected));
  ASSERT_TRUE(smith::equal(input.grad(), smith::ones_like(input)));
}

TEST_F(ModulesTest, Unflatten) {
  // Non-named tensor
  Unflatten unflatten(UnflattenOptions(0, {2, 2}));
  auto output = unflatten->forward(smith::tensor({1, 2, 3, 4}));
  auto expected = smith::tensor({{1, 2}, {3, 4}});
  ASSERT_TRUE(smith::equal(output, expected));

  // Named tensor
  auto make_dimnames = [](std::vector<std::string> names) {
    std::vector<smith::Dimname> dimnames;
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (auto name : names) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      dimnames.push_back(
          smith::Dimname::fromSymbol(smith::Symbol::dimname(name)));
    }
    return dimnames;
  };

  unflatten = Unflatten(UnflattenOptions(
      "B",
      {std::pair<std::string, int64_t>{"B1", 2},
       std::pair<std::string, int64_t>{"B2", 2}}));
  output = unflatten->forward(
      smith::tensor({{1, 2, 3, 4}}).refine_names(make_dimnames({"A", "B"})));
  expected = smith::tensor({{{1, 2}, {3, 4}}})
                 .refine_names(make_dimnames({"A", "B1", "B2"}));
  ASSERT_TRUE(smith::equal(output, expected));
}

TEST_F(ModulesTest, AdaptiveMaxPool1d) {
  AdaptiveMaxPool1d model(3);
  auto x = smith::tensor(
      {{{1, 2, 3, 4, 5}}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::tensor({{{2, 4, 5}}}, smith::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool1dReturnIndices) {
  AdaptiveMaxPool1d model(3);
  auto x = smith::tensor(
      {{{1, 2, 3, 4, 5}}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto [y, indices] = model->forward_with_indices(x);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(smith::allclose(y, smith::tensor({{{2, 4, 5}}}, smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
  ASSERT_TRUE(
      smith::allclose(indices, smith::tensor({{{1, 3, 4}}}, smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dEven) {
  AdaptiveMaxPool2d model(3);
  auto x = smith::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}},
          },
          smith::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dUneven) {
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  auto x = smith::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{25, 27}, {33, 35}, {37, 39}},
          },
          smith::kFloat)));
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesEven) {
  AdaptiveMaxPool2d model(3);
  auto x = smith::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto [y, indices] = model->forward_with_indices(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{31, 33, 34}, {41, 43, 44}, {46, 48, 49}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      indices,
      smith::tensor(
          {
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
              {{6, 8, 9}, {16, 18, 19}, {21, 23, 24}},
          },
          smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool2dReturnIndicesUneven) {
  AdaptiveMaxPool2d model(AdaptiveMaxPool2dOptions({3, 2}));
  auto x = smith::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto [y, indices] = model->forward_with_indices(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{25, 27}, {33, 35}, {37, 39}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));

  ASSERT_EQ(indices.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      indices,
      smith::tensor(
          {
              {{5, 7}, {13, 15}, {17, 19}},
              {{5, 7}, {13, 15}, {17, 19}},
          },
          smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3d) {
  AdaptiveMaxPool3d model(3);
  auto x = smith::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveMaxPool3dReturnIndices) {
  AdaptiveMaxPool3d model(3);
  auto x = smith::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto [y, indices] = model->forward_with_indices(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));

  ASSERT_EQ(indices.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(
      indices,
      smith::tensor(
          {
              {{21, 22, 23}, {25, 26, 27}, {29, 30, 31}},
              {{37, 38, 39}, {41, 42, 43}, {45, 46, 47}},
              {{53, 54, 55}, {57, 58, 59}, {61, 62, 63}},
          },
          smith::kLong)));
  ASSERT_EQ(indices.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool1d) {
  AdaptiveAvgPool1d model(3);
  auto x = smith::tensor(
      {{{1, 2, 3, 4, 5}}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(
      smith::allclose(y, smith::tensor({{{1.5, 3.0, 4.5}}}, smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool2dEven) {
  AdaptiveAvgPool2d model(3);
  auto x = smith::arange(0., 50);
  x.resize_({2, 5, 5}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{3.0, 4.5, 6.0}, {10.5, 12.0, 13.5}, {18.0, 19.5, 21.0}},
              {{28.0, 29.5, 31.0}, {35.5, 37.0, 38.5}, {43.0, 44.5, 46.0}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 3}));
}

TEST_F(ModulesTest, AdaptiveAvgPool2dUneven) {
  AdaptiveAvgPool2d model(AdaptiveAvgPool2dOptions({3, 2}));
  auto x = smith::arange(0., 40);
  x.resize_({2, 5, 4}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{2.5, 4.5}, {8.5, 10.5}, {14.5, 16.5}},
              {{22.5, 24.5}, {28.5, 30.5}, {34.5, 36.5}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 3, 2}));
}

TEST_F(ModulesTest, AdaptiveAvgPool3d) {
  AdaptiveAvgPool3d model(3);
  auto x = smith::arange(0., 64);
  x.resize_({1, 4, 4, 4}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {
              {{10.5, 11.5, 12.5}, {14.5, 15.5, 16.5}, {18.5, 19.5, 20.5}},
              {{26.5, 27.5, 28.5}, {30.5, 31.5, 32.5}, {34.5, 35.5, 36.5}},
              {{42.5, 43.5, 44.5}, {46.5, 47.5, 48.5}, {50.5, 51.5, 52.5}},
          },
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool1d) {
  auto indices = smith::tensor({{{1, 3, 4}}}, smith::kLong);
  auto x = smith::tensor(
      {{{2, 4, 5}}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto model = MaxUnpool1d{3};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(smith::allclose(
      y, smith::tensor({{{0, 2, 0, 4, 5, 0, 0, 0, 0}}}, smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 9}));

  indices = smith::tensor({{{1, 3, 4}}}, smith::kLong);
  x = smith::tensor(
      {{{2, 4, 5}}}, smith::dtype(smith::kFloat).requires_grad(true));
  model = MaxUnpool1d{MaxUnpool1dOptions(3).stride(2).padding(1)};
  y = model->forward(x, indices, std::vector<int64_t>({1, 1, 5}));

  ASSERT_EQ(y.dim(), 3);
  ASSERT_TRUE(
      smith::allclose(y, smith::tensor({{{0, 2, 0, 4, 5}}}, smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 5}));
}

TEST_F(ModulesTest, MaxPool1d_MaxUnpool1d) {
  MaxPool1d pool{MaxPool1dOptions(2).stride(2)};
  MaxUnpool1d unpool{MaxUnpool1dOptions(2).stride(2)};
  auto input = smith::tensor({{{1, 2, 3, 4, 5, 6, 7, 8}}}, smith::kFloat);
  auto [output, indices] = pool->forward_with_indices(input);
  ASSERT_TRUE(smith::allclose(
      unpool(output, indices),
      smith::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}}, smith::kFloat)));

  // Example showcasing the use of output_size
  input = smith::tensor({{{1, 2, 3, 4, 5, 6, 7, 8, 9}}}, smith::kFloat);
  std::tie(output, indices) = pool->forward_with_indices(input);
  ASSERT_TRUE(smith::allclose(
      unpool(output, indices, input.sizes().vec()),
      smith::tensor({{{0, 2, 0, 4, 0, 6, 0, 8, 0}}}, smith::kFloat)));
  ASSERT_TRUE(smith::allclose(
      unpool(output, indices),
      smith::tensor({{{0, 2, 0, 4, 0, 6, 0, 8}}}, smith::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool2d) {
  auto indices = smith::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}}},
      smith::kLong);
  auto x = smith::tensor(
      {{{{6, 8, 9}, {16, 18, 19}, {21, 23, 24}}},
       {{{31, 33, 34}, {41, 43, 44}, {46, 48, 49}}}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto model = MaxUnpool2d{MaxUnpool2dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 4);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {{{{0, 0, 0, 0, 0},
             {0, 6, 0, 8, 9},
             {0, 0, 0, 0, 0},
             {0, 16, 0, 18, 19},
             {0, 21, 0, 23, 24}}},
           {{{0, 0, 0, 0, 0},
             {0, 31, 0, 33, 34},
             {0, 0, 0, 0, 0},
             {0, 41, 0, 43, 44},
             {0, 46, 0, 48, 49}}}},
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({2, 1, 5, 5}));
}

TEST_F(ModulesTest, MaxPool2d_MaxUnpool2d) {
  MaxPool2d pool{MaxPool2dOptions(2).stride(2)};
  MaxUnpool2d unpool{MaxUnpool2dOptions(2).stride(2)};
  auto input = smith::tensor(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}}},
      smith::kFloat);
  auto [output, indices] = pool->forward_with_indices(input);
  ASSERT_TRUE(smith::allclose(
      unpool(output, indices),
      smith::tensor(
          {{{{0, 0, 0, 0}, {0, 6, 0, 8}, {0, 0, 0, 0}, {0, 14, 0, 16}}}},
          smith::kFloat)));

  ASSERT_TRUE(smith::allclose(
      unpool(output, indices, std::vector<int64_t>{1, 1, 5, 5}),
      smith::tensor(
          {{{{0, 0, 0, 0, 0},
             {6, 0, 8, 0, 0},
             {0, 0, 0, 14, 0},
             {16, 0, 0, 0, 0},
             {0, 0, 0, 0, 0}}}},
          smith::kFloat)));
}

TEST_F(ModulesTest, MaxUnpool3d) {
  auto indices = smith::tensor({{{{{26}}}}}, smith::kLong);
  auto x = smith::tensor(
      {{{{{26}}}}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto model = MaxUnpool3d{3};
  auto y = model->forward(x, indices);

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {{{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
             {{0, 0, 0}, {0, 0, 0}, {0, 0, 26}}}}},
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 3, 3, 3}));
}

TEST_F(ModulesTest, MaxUnpool3dOutputSize) {
  auto indices = smith::tensor(
      {{{{{21, 23}, {29, 31}}, {{53, 55}, {61, 63}}}}}, smith::kLong);
  auto x = smith::tensor(
      {{{{{21, 23}, {29, 31}}, {{53, 55}, {61, 63}}}}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto model = MaxUnpool3d{MaxUnpool3dOptions(3).stride(2).padding(1)};
  auto y = model->forward(x, indices, std::vector<int64_t>({1, 1, 4, 4, 4}));

  ASSERT_EQ(y.dim(), 5);
  ASSERT_TRUE(smith::allclose(
      y,
      smith::tensor(
          {{{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
             {{0, 0, 0, 0}, {0, 21, 0, 23}, {0, 0, 0, 0}, {0, 29, 0, 31}},
             {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
             {{0, 0, 0, 0}, {0, 53, 0, 55}, {0, 0, 0, 0}, {0, 61, 0, 63}}}}},
          smith::kFloat)));
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({1, 1, 4, 4, 4}));
}

TEST_F(ModulesTest, MaxPool3d_MaxUnpool3d) {
  MaxPool3d pool{MaxPool3dOptions(3).stride(2)};
  MaxUnpool3d unpool{MaxUnpool3dOptions(3).stride(2)};
  auto input = smith::randn({20, 16, 51, 33, 15});
  auto [output, indices] = pool->forward_with_indices(input);
  auto unpooled_output = unpool(output, indices);
  ASSERT_EQ(
      unpooled_output.sizes(), std::vector<int64_t>({20, 16, 51, 33, 15}));
}

TEST_F(ModulesTest, Linear) {
  {
    Linear model(5, 2);
    auto x = smith::randn({10, 5}, smith::requires_grad());
    auto y = model(x);
    smith::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.size(0), 10);
    ASSERT_EQ(y.size(1), 2);

    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    auto y_exp = smith::addmm(model->bias, x, model->weight.t());
    ASSERT_TRUE(smith::allclose(y, y_exp));
  }
  {
    Linear model(LinearOptions(5, 2).bias(false));
    auto x = smith::randn({10, 5}, smith::requires_grad());
    auto y = model(x);
    smith::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 2);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.size(0), 10);
    ASSERT_EQ(y.size(1), 2);

    ASSERT_EQ(model->weight.grad().numel(), 2 * 5);

    auto y_exp = smith::mm(x, model->weight.t());
    ASSERT_TRUE(smith::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, LocalResponseNorm) {
  {
    LocalResponseNorm model(LocalResponseNormOptions(2));
    const auto x =
        smith::arange(100., 136, smith::requires_grad()).reshape({2, 3, 3, 2});
    auto y = model(x);
    const auto y_exp = smith::tensor(
        {{{{73.7788, 74.1462}, {74.5031, 74.8572}, {75.2010, 75.5420}},

          {{61.6057, 61.7227}, {61.8347, 61.9418}, {62.0441, 62.1418}},

          {{62.2349, 62.3235}, {62.4077, 62.4877}, {62.5635, 62.6353}}},

         {{{79.3915, 79.6491}, {79.8978, 80.1446}, {80.3827, 80.6190}},

          {{63.0317, 63.0742}, {63.1135, 63.1496}, {63.1826, 63.2126}},

          {{63.2396, 63.2637}, {63.2850, 63.3036}, {63.3195, 63.3328}}}},
        smith::kFloat);
    smith::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(y.ndimension(), 4);
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.sizes(), x.sizes());
    ASSERT_TRUE(smith::allclose(y, y_exp, 1e-4, 1e-7));
  }
}

TEST_F(ModulesTest, LayerNorm) {
  LayerNorm model(LayerNormOptions({2, 2}).eps(2e-5));
  auto x = smith::randn({2, 2}, smith::requires_grad());
  auto y = model(x);
  auto y_exp = smith::layer_norm(x, {2, 2}, model->weight, model->bias, 2e-5);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  for (const auto i : c10::irange(2)) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 2 * 2);
  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, GroupNorm) {
  GroupNorm model(GroupNormOptions(2, 2).eps(2e-5));
  auto x = smith::randn({2, 2}, smith::requires_grad());
  auto y = model(x);
  auto y_exp = smith::group_norm(x, 2, model->weight, model->bias, 2e-5);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  for (const auto i : c10::irange(2)) {
    ASSERT_EQ(y.size(i), 2);
  }

  ASSERT_EQ(model->weight.grad().numel(), 2);
  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, Bilinear) {
  Bilinear model(5, 3, 2);
  auto x1 = smith::randn({10, 5}, smith::requires_grad());
  auto x2 = smith::randn({10, 3}, smith::requires_grad());
  auto y = model(x1, x2);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5 * 3);
}

TEST_F(ModulesTest, Fold) {
  {
    Fold model(FoldOptions({3, 2}, {2, 2}));
    auto input = smith::ones({1, 3 * 2 * 2, 2}, smith::requires_grad());
    auto output = model(input);
    auto expected = smith::tensor(
        {{{{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
          {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}},
          {{1.0, 1.0}, {2.0, 2.0}, {1.0, 1.0}}}},
        smith::kFloat);
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 3, 2}));
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    Fold model(FoldOptions({8, 8}, {3, 3}));
    ASSERT_THROWS_WITH(
        model(smith::randn({1, 3, 16, 16})),
        "Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported (got 4D)");
  }
}

TEST_F(ModulesTest, Unfold) {
  {
    Unfold model(UnfoldOptions({2, 2}).padding(1).stride(2));
    auto input =
        smith::arange(2., 14, smith::requires_grad()).view({1, 2, 2, 3});
    auto output = model(input);
    auto expected = smith::tensor(
        {{{0.0, 0.0, 0.0, 6.0},
          {0.0, 0.0, 5.0, 7.0},
          {0.0, 3.0, 0.0, 0.0},
          {2.0, 4.0, 0.0, 0.0},
          {0.0, 0.0, 0.0, 12.0},
          {0.0, 0.0, 11.0, 13.0},
          {0.0, 9.0, 0.0, 0.0},
          {8.0, 10.0, 0.0, 0.0}}},
        smith::kFloat);
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 8, 4}));
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    // input wrong dimension
    Unfold model(UnfoldOptions({2, 4}));
    ASSERT_THROWS_WITH(
        model(smith::randn({1, 5, 2})),
        "Input Error: Only 4D input Tensors are supported (got 3D)");
  }
  {
    // calculated output shape is too small
    Unfold model(UnfoldOptions({2, 3}));
    ASSERT_THROWS_WITH(
        model(smith::randn({1, 2, 2, 2})),
        "Given input with spatial size (2, 2), kernel_size=(2, 3), "
        "dilation=(1, 1), padding=(0, 0), calculated shape of the array of "
        "sliding blocks as (1, 0), but its components must be at least one.");
  }
}

TEST_F(ModulesTest, SimpleContainer) {
  auto model = std::make_shared<SimpleContainer>();
  auto l1 = model->add(Linear(10, 3), "l1");
  auto l2 = model->add(Linear(3, 5), "l2");
  auto l3 = model->add(Linear(5, 100), "l3");

  auto x = smith::randn({1000, 10}, smith::requires_grad());
  x = l1(x).clamp_min(0);
  x = l2(x).clamp_min(0);
  x = l3(x).clamp_min(0);

  x.backward(smith::ones_like(x));
  ASSERT_EQ(x.ndimension(), 2);
  ASSERT_EQ(x.size(0), 1000);
  ASSERT_EQ(x.size(1), 100);
  ASSERT_EQ(x.min().item<float>(), 0);
}

TEST_F(ModulesTest, EmbeddingBasic) {
  const int64_t dict_size = 10;
  Embedding model(dict_size, 2);
  ASSERT_TRUE(model->named_parameters().contains("weight"));
  ASSERT_EQ(model->weight.ndimension(), 2);
  ASSERT_EQ(model->weight.size(0), dict_size);
  ASSERT_EQ(model->weight.size(1), 2);

  // Cannot get gradients to change indices (input) - only for embedding
  // params
  auto x = smith::full({10}, dict_size - 1, smith::kInt64);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * dict_size);
}

TEST_F(ModulesTest, EmbeddingList) {
  Embedding model(6, 4);
  auto x = smith::full({2, 3}, 5, smith::kInt64);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.size(0), 2);
  ASSERT_EQ(y.size(1), 3);
  ASSERT_EQ(y.size(2), 4);
}

TEST_F(ModulesTest, EmbeddingFromPretrained) {
  auto weight = smith::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  Embedding embedding = smith::nn::Embedding::from_pretrained(weight);
  auto input = smith::tensor({1}, smith::kLong);
  ASSERT_TRUE(smith::allclose(
      embedding(input), smith::tensor({4.0000, 5.1000, 6.3000})));
}

TEST_F(ModulesTest, EmbeddingBagFromPretrained) {
  auto weight = smith::tensor({{1., 2.3, 3.}, {4., 5.1, 6.3}});
  EmbeddingBag embeddingbag = smith::nn::EmbeddingBag::from_pretrained(weight);
  auto input = smith::zeros({{1, 2}}, smith::kLong);
  input[0] = smith::tensor({1, 0});
  ASSERT_TRUE(smith::allclose(
      embeddingbag(input), smith::tensor({2.5000, 3.7000, 4.6500})));
}

TEST_F(ModulesTest, AlphaDropout) {
  AlphaDropout alpha_dropout(0.5);
  smith::Tensor x = smith::ones(100, smith::requires_grad());
  smith::Tensor y = alpha_dropout(x);

  y.backward(smith::ones_like(y));

  ASSERT_EQ(y.ndimension(), 1);
  ASSERT_EQ(y.size(0), 100);
  ASSERT_LT(y.sum().item<float>(), 130); // Probably
  ASSERT_GT(y.sum().item<float>(), 40); // Probably

  alpha_dropout->eval();
  y = alpha_dropout(x);

  ASSERT_EQ(y.sum().item<float>(), 100);
}

TEST_F(ModulesTest, FeatureAlphaDropout) {
  FeatureAlphaDropout feature_alpha_dropout(0.5);
  smith::Tensor x = smith::ones({10, 10}, smith::requires_grad());
  smith::Tensor y = feature_alpha_dropout(x);

  y.backward(smith::ones_like(y));

  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 10);
  ASSERT_LT(y.sum().item<float>(), 130); // Probably
  ASSERT_GT(y.sum().item<float>(), 40); // Probably

  feature_alpha_dropout->eval();
  y = feature_alpha_dropout(x);

  ASSERT_EQ(y.sum().item<float>(), 100);
}

TEST_F(ModulesTest, Dropout) {
  for (const auto inplace : {false, true}) {
    Dropout dropout(DropoutOptions(0.5).inplace(inplace));
    smith::Tensor x = smith::ones(100);
    if (!inplace) {
      x.requires_grad_(true);
    }
    smith::Tensor y = dropout(x);

    ASSERT_EQ(y.ndimension(), 1);
    ASSERT_EQ(y.size(0), 100);
    ASSERT_LT(y.sum().item<float>(), 130); // Probably
    ASSERT_GT(y.sum().item<float>(), 70); // Probably
    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      y.backward(smith::ones_like(y));
    }

    dropout->eval();
    y = dropout(smith::ones(100));
    ASSERT_EQ(y.sum().item<float>(), 100);
  }
}

TEST_F(ModulesTest, Dropout2d) {
  auto p = 0.5;
  for (const auto inplace : {false, true}) {
    Dropout2d dropout(Dropout2dOptions(p).inplace(inplace));
    smith::Tensor x = smith::empty({50, 50, 2, 2}).fill_(1 - p);
    if (!inplace) {
      x.requires_grad_(true);
    }
    smith::Tensor y = dropout(x);

    ASSERT_EQ(y.ndimension(), 4);
    ASSERT_EQ(y.size(0), 50);
    ASSERT_EQ(y.size(1), 50);
    ASSERT_EQ(y.size(2), 2);
    ASSERT_EQ(y.size(3), 2);
    ASSERT_LT((y.mean() - (1 - p)).abs().item<float>(), 0.05);

    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      y.backward(smith::ones_like(y));
    }

    dropout->eval();
    y = dropout(smith::ones({2, 2, 10, 10}));
    ASSERT_EQ(y.sum().item<float>(), 400);
  }
}

TEST_F(ModulesTest, Dropout3d) {
  for (const auto inplace : {false, true}) {
    auto p = 0.5;
    Dropout3d dropout(Dropout3dOptions(p).inplace(inplace));
    smith::Tensor x = smith::empty({50, 50, 2, 2, 2}).fill_(1 - p);
    if (!inplace) {
      x.requires_grad_(true);
    }
    smith::Tensor y = dropout(x);

    ASSERT_EQ(y.ndimension(), 5);
    ASSERT_EQ(y.size(0), 50);
    ASSERT_EQ(y.size(1), 50);
    ASSERT_EQ(y.size(2), 2);
    ASSERT_EQ(y.size(3), 2);
    ASSERT_EQ(y.size(4), 2);
    ASSERT_LT((y.mean() - (1 - p)).abs().item<float>(), 0.05);

    if (inplace) {
      ASSERT_TRUE(y.allclose(x));
    } else {
      y.backward(smith::ones_like(y));
    }

    dropout->eval();
    y = dropout(smith::ones({4, 4, 5, 5}));
    ASSERT_EQ(y.sum().item<float>(), 400);
  }
}

TEST_F(ModulesTest, Parameters) {
  auto model = std::make_shared<NestedModel>();
  auto parameters = model->named_parameters();
  ASSERT_EQ(parameters["param"].size(0), 3);
  ASSERT_EQ(parameters["param"].size(1), 2);
  ASSERT_EQ(parameters["param"].size(2), 21);
  ASSERT_EQ(parameters["l1.bias"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(0), 20);
  ASSERT_EQ(parameters["l1.weight"].size(1), 5);
  ASSERT_EQ(parameters["test.l1.bias"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(0), 3);
  ASSERT_EQ(parameters["test.l1.weight"].size(1), 10);
  ASSERT_EQ(parameters["test.l2.bias"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(0), 5);
  ASSERT_EQ(parameters["test.l2.weight"].size(1), 3);
  ASSERT_EQ(parameters["test.l3.bias"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(0), 100);
  ASSERT_EQ(parameters["test.l3.weight"].size(1), 5);
}

TEST_F(ModulesTest, FunctionalCallsSuppliedFunction) {
  bool was_called = false;
  auto functional = Functional([&was_called](smith::Tensor input) {
    was_called = true;
    return input;
  });
  auto output = functional(smith::ones(5, smith::requires_grad()));
  ASSERT_TRUE(was_called);
  ASSERT_TRUE(output.equal(smith::ones(5, smith::requires_grad())));

  was_called = false;
  // Use the call operator overload here.
  output = functional(smith::ones(5, smith::requires_grad()));
  ASSERT_TRUE(was_called);
  ASSERT_TRUE(output.equal(smith::ones(5, smith::requires_grad())));
}

TEST_F(ModulesTest, FunctionalWithSmithFunction) {
  auto functional = Functional(smith::relu);
  ASSERT_EQ(functional(smith::ones({})).item<float>(), 1);
  ASSERT_EQ(functional(smith::ones({})).item<float>(), 1);
  ASSERT_EQ(functional(smith::ones({}) * -1).item<float>(), 0);
}

TEST_F(ModulesTest, FunctionalArgumentBinding) {
  auto functional =
      Functional(smith::elu, /*alpha=*/1, /*scale=*/0, /*input_scale=*/1);
  ASSERT_EQ(functional(smith::ones({})).item<float>(), 0);
}

TEST_F(ModulesTest, BatchNorm1dStateful) {
  BatchNorm1d bn(5);

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm1dStateless) {
  BatchNorm1d bn(
      BatchNorm1dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm1d) {
  BatchNorm1d bn(5);
  bn->eval();

  auto input = smith::arange(2. * 5 * 2).view({2, 5, 2}).requires_grad_();
  auto output = bn->forward(input);
  auto expected = smith::tensor(
      {{{0.0000, 1.0000},
        {2.0000, 3.0000},
        {4.0000, 5.0000},
        {6.0000, 7.0000},
        {8.0000, 9.0000}},
       {{10.0000, 10.9999},
        {11.9999, 12.9999},
        {13.9999, 14.9999},
        {15.9999, 16.9999},
        {17.9999, 18.9999}}});
  ASSERT_TRUE(output.allclose(expected));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, BatchNorm2dStateful) {
  BatchNorm2d bn(5);

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm2dStateless) {
  BatchNorm2d bn(
      BatchNorm2dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm2d) {
  BatchNorm2d bn(5);
  bn->eval();

  auto input =
      smith::arange(2. * 5 * 2 * 2).view({2, 5, 2, 2}).requires_grad_();
  auto output = bn->forward(input);
  auto expected = smith::tensor(
      {{{{0.0000, 1.0000}, {2.0000, 3.0000}},
        {{4.0000, 5.0000}, {6.0000, 7.0000}},
        {{8.0000, 9.0000}, {10.0000, 10.9999}},
        {{11.9999, 12.9999}, {13.9999, 14.9999}},
        {{15.9999, 16.9999}, {17.9999, 18.9999}}},
       {{{19.9999, 20.9999}, {21.9999, 22.9999}},
        {{23.9999, 24.9999}, {25.9999, 26.9999}},
        {{27.9999, 28.9999}, {29.9998, 30.9998}},
        {{31.9998, 32.9998}, {33.9998, 34.9998}},
        {{35.9998, 36.9998}, {37.9998, 38.9998}}}});
  ASSERT_TRUE(output.allclose(expected));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, BatchNorm3dStateful) {
  BatchNorm3d bn(5);

  ASSERT_TRUE(bn->options.track_running_stats());

  ASSERT_TRUE(bn->running_mean.defined());
  ASSERT_EQ(bn->running_mean.dim(), 1);
  ASSERT_EQ(bn->running_mean.size(0), 5);

  ASSERT_TRUE(bn->running_var.defined());
  ASSERT_EQ(bn->running_var.dim(), 1);
  ASSERT_EQ(bn->running_var.size(0), 5);

  ASSERT_TRUE(bn->num_batches_tracked.defined());
  ASSERT_EQ(bn->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(bn->options.affine());

  ASSERT_TRUE(bn->weight.defined());
  ASSERT_EQ(bn->weight.dim(), 1);
  ASSERT_EQ(bn->weight.size(0), 5);

  ASSERT_TRUE(bn->bias.defined());
  ASSERT_EQ(bn->bias.dim(), 1);
  ASSERT_EQ(bn->bias.size(0), 5);
}

TEST_F(ModulesTest, BatchNorm3dStateless) {
  BatchNorm3d bn(
      BatchNorm3dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(bn->running_mean.defined());
  ASSERT_FALSE(bn->running_var.defined());
  ASSERT_FALSE(bn->num_batches_tracked.defined());
  ASSERT_FALSE(bn->weight.defined());
  ASSERT_FALSE(bn->bias.defined());
}

TEST_F(ModulesTest, BatchNorm3d) {
  BatchNorm3d bn(5);
  bn->eval();

  auto input =
      smith::arange(2. * 5 * 2 * 2 * 2).view({2, 5, 2, 2, 2}).requires_grad_();
  auto output = bn->forward(input);
  auto expected = smith::tensor(
      {{{{{0.0000, 1.0000}, {2.0000, 3.0000}},
         {{4.0000, 5.0000}, {6.0000, 7.0000}}},
        {{{8.0000, 9.0000}, {10.0000, 10.9999}},
         {{11.9999, 12.9999}, {13.9999, 14.9999}}},
        {{{15.9999, 16.9999}, {17.9999, 18.9999}},
         {{19.9999, 20.9999}, {21.9999, 22.9999}}},
        {{{23.9999, 24.9999}, {25.9999, 26.9999}},
         {{27.9999, 28.9999}, {29.9998, 30.9998}}},
        {{{31.9998, 32.9998}, {33.9998, 34.9998}},
         {{35.9998, 36.9998}, {37.9998, 38.9998}}}},
       {{{{39.9998, 40.9998}, {41.9998, 42.9998}},
         {{43.9998, 44.9998}, {45.9998, 46.9998}}},
        {{{47.9998, 48.9998}, {49.9997, 50.9997}},
         {{51.9997, 52.9997}, {53.9997, 54.9997}}},
        {{{55.9997, 56.9997}, {57.9997, 58.9997}},
         {{59.9997, 60.9997}, {61.9997, 62.9997}}},
        {{{63.9997, 64.9997}, {65.9997, 66.9997}},
         {{67.9997, 68.9997}, {69.9996, 70.9996}}},
        {{{71.9996, 72.9996}, {73.9996, 74.9996}},
         {{75.9996, 76.9996}, {77.9996, 78.9996}}}}});
  ASSERT_TRUE(output.allclose(expected));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, InstanceNorm1dStateful) {
  InstanceNorm1d instance_norm(
      InstanceNorm1dOptions(5).track_running_stats(true).affine(true));

  ASSERT_TRUE(instance_norm->options.track_running_stats());

  ASSERT_TRUE(instance_norm->running_mean.defined());
  ASSERT_EQ(instance_norm->running_mean.dim(), 1);
  ASSERT_EQ(instance_norm->running_mean.size(0), 5);

  ASSERT_TRUE(instance_norm->running_var.defined());
  ASSERT_EQ(instance_norm->running_var.dim(), 1);
  ASSERT_EQ(instance_norm->running_var.size(0), 5);

  ASSERT_TRUE(instance_norm->num_batches_tracked.defined());
  ASSERT_EQ(instance_norm->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(instance_norm->options.affine());

  ASSERT_TRUE(instance_norm->weight.defined());
  ASSERT_EQ(instance_norm->weight.dim(), 1);
  ASSERT_EQ(instance_norm->weight.size(0), 5);

  ASSERT_TRUE(instance_norm->bias.defined());
  ASSERT_EQ(instance_norm->bias.dim(), 1);
  ASSERT_EQ(instance_norm->bias.size(0), 5);
}

TEST_F(ModulesTest, InstanceNorm1dStateless) {
  InstanceNorm1d instance_norm(
      InstanceNorm1dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(instance_norm->running_mean.defined());
  ASSERT_FALSE(instance_norm->running_var.defined());
  ASSERT_FALSE(instance_norm->num_batches_tracked.defined());
  ASSERT_FALSE(instance_norm->weight.defined());
  ASSERT_FALSE(instance_norm->bias.defined());
}

TEST_F(ModulesTest, InstanceNorm1d) {
  InstanceNorm1d instance_norm(5);
  instance_norm->eval();

  auto input = smith::arange(2. * 5 * 2).view({2, 5, 2}).requires_grad_();
  auto output = instance_norm->forward(input);
  auto expected = smith::tensor(
      {{{-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000}},
       {{-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000},
        {-1.0000, 1.0000}}});
  ASSERT_TRUE(output.allclose(expected, 1e-3));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, InstanceNorm2dStateful) {
  InstanceNorm2d instance_norm(
      InstanceNorm2dOptions(5).track_running_stats(true).affine(true));

  ASSERT_TRUE(instance_norm->options.track_running_stats());

  ASSERT_TRUE(instance_norm->running_mean.defined());
  ASSERT_EQ(instance_norm->running_mean.dim(), 1);
  ASSERT_EQ(instance_norm->running_mean.size(0), 5);

  ASSERT_TRUE(instance_norm->running_var.defined());
  ASSERT_EQ(instance_norm->running_var.dim(), 1);
  ASSERT_EQ(instance_norm->running_var.size(0), 5);

  ASSERT_TRUE(instance_norm->num_batches_tracked.defined());
  ASSERT_EQ(instance_norm->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(instance_norm->options.affine());

  ASSERT_TRUE(instance_norm->weight.defined());
  ASSERT_EQ(instance_norm->weight.dim(), 1);
  ASSERT_EQ(instance_norm->weight.size(0), 5);

  ASSERT_TRUE(instance_norm->bias.defined());
  ASSERT_EQ(instance_norm->bias.dim(), 1);
  ASSERT_EQ(instance_norm->bias.size(0), 5);
}

TEST_F(ModulesTest, InstanceNorm2dStateless) {
  InstanceNorm2d instance_norm(
      InstanceNorm2dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(instance_norm->running_mean.defined());
  ASSERT_FALSE(instance_norm->running_var.defined());
  ASSERT_FALSE(instance_norm->num_batches_tracked.defined());
  ASSERT_FALSE(instance_norm->weight.defined());
  ASSERT_FALSE(instance_norm->bias.defined());
}

TEST_F(ModulesTest, InstanceNorm2d) {
  InstanceNorm2d instance_norm(5);
  instance_norm->eval();

  auto input =
      smith::arange(2. * 5 * 2 * 2).view({2, 5, 2, 2}).requires_grad_();
  auto output = instance_norm->forward(input);
  auto expected = smith::tensor(
      {{{{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}}},
       {{{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}},
        {{-1.3416, -0.4472}, {0.4472, 1.3416}}}});
  ASSERT_TRUE(output.allclose(expected, 1e-3));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, InstanceNorm3dStateful) {
  InstanceNorm3d instance_norm(
      InstanceNorm3dOptions(5).track_running_stats(true).affine(true));

  ASSERT_TRUE(instance_norm->options.track_running_stats());

  ASSERT_TRUE(instance_norm->running_mean.defined());
  ASSERT_EQ(instance_norm->running_mean.dim(), 1);
  ASSERT_EQ(instance_norm->running_mean.size(0), 5);

  ASSERT_TRUE(instance_norm->running_var.defined());
  ASSERT_EQ(instance_norm->running_var.dim(), 1);
  ASSERT_EQ(instance_norm->running_var.size(0), 5);

  ASSERT_TRUE(instance_norm->num_batches_tracked.defined());
  ASSERT_EQ(instance_norm->num_batches_tracked.dim(), 0);

  ASSERT_TRUE(instance_norm->options.affine());

  ASSERT_TRUE(instance_norm->weight.defined());
  ASSERT_EQ(instance_norm->weight.dim(), 1);
  ASSERT_EQ(instance_norm->weight.size(0), 5);

  ASSERT_TRUE(instance_norm->bias.defined());
  ASSERT_EQ(instance_norm->bias.dim(), 1);
  ASSERT_EQ(instance_norm->bias.size(0), 5);
}

TEST_F(ModulesTest, InstanceNorm3dStateless) {
  InstanceNorm3d instance_norm(
      InstanceNorm3dOptions(5).track_running_stats(false).affine(false));

  ASSERT_FALSE(instance_norm->running_mean.defined());
  ASSERT_FALSE(instance_norm->running_var.defined());
  ASSERT_FALSE(instance_norm->num_batches_tracked.defined());
  ASSERT_FALSE(instance_norm->weight.defined());
  ASSERT_FALSE(instance_norm->bias.defined());
}

TEST_F(ModulesTest, InstanceNorm3d) {
  InstanceNorm3d instance_norm(5);
  instance_norm->eval();

  auto input =
      smith::arange(2. * 5 * 2 * 2 * 2).view({2, 5, 2, 2, 2}).requires_grad_();
  auto output = instance_norm->forward(input);
  auto expected = smith::tensor(
      {{{{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}}},
       {{{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}},
        {{{-1.5275, -1.0911}, {-0.6547, -0.2182}},
         {{0.2182, 0.6547}, {1.0911, 1.5275}}}}});
  ASSERT_TRUE(output.allclose(expected, 1e-3));
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, Linear_CUDA) {
  Linear model(5, 2);
  model->to(smith::kCUDA);
  auto x =
      smith::randn({10, 5}, smith::device(smith::kCUDA).requires_grad(true));
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5);
}

TEST_F(ModulesTest, Linear2_CUDA) {
  Linear model(5, 2);
  model->to(smith::kCUDA);
  model->to(smith::kCPU);
  auto x = smith::randn({10, 5}, smith::requires_grad());
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(y.ndimension(), 2);
  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_EQ(y.size(0), 10);
  ASSERT_EQ(y.size(1), 2);

  ASSERT_EQ(model->weight.grad().numel(), 2 * 5);
}

TEST_F(ModulesTest, L1Loss) {
  L1Loss loss;
  auto input = smith::randn({5, 6}, smith::requires_grad());
  auto target = smith::empty({5, 6}).random_(2);
  auto output = loss->forward(smith::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), std::vector<int64_t>());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MSELoss) {
  MSELoss loss;
  auto input = smith::randn({5, 6}, smith::requires_grad());
  auto target = smith::empty({5, 6}).random_(2);
  auto output = loss->forward(smith::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), smith::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, BCELoss) {
  BCELoss loss;
  auto input = smith::randn({5, 6}, smith::requires_grad());
  auto target = smith::empty({5, 6}).random_(2);
  auto output = loss->forward(smith::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), smith::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, KLDivLoss) {
  KLDivLoss loss;
  auto input = smith::randn({5, 6}, smith::requires_grad());
  auto target = smith::empty({5, 6}).random_(2);
  auto output = loss->forward(smith::sigmoid(input), target);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(output.sizes(), smith::IntArrayRef());
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HingeEmbeddingLoss) {
  HingeEmbeddingLoss loss(HingeEmbeddingLossOptions().margin(2));
  auto input = smith::tensor(
      {{2, 22, 4}, {20, 10, 0}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({{2, 6, 4}, {1, 10, 0}}, smith::kFloat);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({10}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiMarginLoss) {
  auto weight = smith::tensor({0.3, 0.3, 0.4}, smith::kFloat);
  MultiMarginLoss loss(MultiMarginLossOptions().margin(2).weight(weight));
  auto input = smith::tensor(
      {{0.2, 0.2, 0.6}, {0.1, 0.8, 0.1}, {0.9, 0.09, 0.01}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({2, 1, 0}, smith::kLong);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({0.305556}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, CosineEmbeddingLoss) {
  CosineEmbeddingLoss cos(CosineEmbeddingLossOptions().margin(0.5));
  auto input1 = smith::tensor(
      {{2, 3, 4}, {6, 2, 4}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto input2 = smith::tensor(
      {{2, 3, 5}, {9, 12, 0}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({1, -1});
  auto output = cos(input1, input2, target);
  auto expected = smith::tensor({0.1004}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-4));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
  ASSERT_EQ(input2.sizes(), input2.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossDefaultOptions) {
  SmoothL1Loss loss;
  auto input = smith::tensor(
      {0.1, 1.2, 4.7}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0., 1., 5.}, smith::kFloat);
  auto output = loss(input, target);
  auto expected = smith::tensor(0.0233335, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HuberLossDefaultOptions) {
  HuberLoss loss;
  auto input = smith::tensor(
      {0.1, 1.2, 4.7}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0., 1., 5.}, smith::kFloat);
  auto output = loss(input, target);
  auto expected = smith::tensor(0.0233335, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelMarginLossDefaultOptions) {
  MultiLabelMarginLoss loss;
  auto input = smith::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({{3, 0, -1, 1}}, smith::kLong);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({0.8500}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossNoReduction) {
  SmoothL1Loss loss(/*reduction=*/smith::kNone);
  auto input = smith::tensor(
      {0.1, 1.2, 4.7}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0., 1., 5.}, smith::kFloat);
  auto output = loss(input, target);
  auto expected = smith::tensor({0.005, 0.02, 0.045}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HuberLossNoReduction) {
  HuberLoss loss(/*reduction=*/smith::kNone);
  auto input = smith::tensor(
      {0.1, 1.2, 4.7}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0., 1., 5.}, smith::kFloat);
  auto output = loss(input, target);
  auto expected = smith::tensor({0.005, 0.02, 0.045}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelMarginLossNoReduction) {
  MultiLabelMarginLoss loss(smith::kNone);
  auto input = smith::tensor(
      {{0.1, 0.2, 0.4, 0.8}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({{3, 0, -1, 1}}, smith::kLong);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({0.8500}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SmoothL1LossBeta) {
  auto options = SmoothL1LossOptions().beta(0.2);
  SmoothL1Loss loss(options);
  auto input = smith::tensor(
      {0.1, 1.2, 4.7}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0., 1., 5.}, smith::kFloat);
  auto output = loss(input, target);
  auto expected = smith::tensor(0.108333, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, HuberLossDelta) {
  auto options = HuberLossOptions().delta(0.2);
  HuberLoss loss(options);
  auto input = smith::tensor(
      {0.1, 1.2, 4.7}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0., 1., 5.}, smith::kFloat);
  auto output = loss(input, target);
  auto expected = smith::tensor(0.0216666, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, TripletMarginLoss) {
  TripletMarginLoss loss(TripletMarginLossOptions().margin(1.0));
  auto anchor = smith::tensor(
      {{3., 3.}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto positive = smith::tensor(
      {{2., 2.}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto negative = smith::tensor(
      {{0., 0.}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto output = loss->forward(anchor, positive, negative);
  auto expected = smith::tensor({0.}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
}

TEST_F(ModulesTest, TripletMarginWithDistanceLossDefaultParity) {
  // Check that if we use smith::pairwise_distance with the default
  // TripletMarginLoss options as our distance function, the outputs
  // are equal (i.e., equal under defaults).

  std::vector<TripletMarginWithDistanceLossOptions::reduction_t> reductions = {
      smith::kSum, smith::kMean, smith::kNone};
  std::vector<float> margins = {0.5, 1.0, 1.5};
  std::vector<bool> swaps = {true, false};

  for (auto& reduction : reductions) {
    for (auto& margin : margins) {
      for (const auto swap : swaps) {
        auto anchor = smith::randn(
            {100, 128}, smith::dtype(smith::kFloat).requires_grad(true));
        auto positive = smith::randn(
            {100, 128}, smith::dtype(smith::kFloat).requires_grad(true));
        auto negative = smith::randn(
            {100, 128}, smith::dtype(smith::kFloat).requires_grad(true));

        auto basicOptions =
            TripletMarginLossOptions().reduction(reduction).margin(margin).swap(
                swap);
        auto distanceOptions = TripletMarginWithDistanceLossOptions()
                                   .reduction(reduction)
                                   .margin(margin)
                                   .swap(swap);
        TripletMarginLoss basicLoss(basicOptions);
        TripletMarginWithDistanceLoss distanceLoss(distanceOptions);

        auto basicOutput = basicLoss->forward(anchor, positive, negative);
        auto distanceOutput = distanceLoss->forward(anchor, positive, negative);
        auto basicOperatorOutput = basicLoss(anchor, positive, negative);
        auto distanceOperatorOutput = distanceLoss(anchor, positive, negative);

        ASSERT_TRUE(distanceOutput.allclose(basicOutput, 1e-6, 1e-6));
        ASSERT_TRUE(
            distanceOperatorOutput.allclose(distanceOutput, 1e-6, 1e-6));
        ASSERT_TRUE(
            distanceOperatorOutput.allclose(basicOperatorOutput, 1e-6, 1e-6));

        // handle for smith::kNone reduction
        auto sum = distanceOutput.sum();
        sum.backward();
        ASSERT_EQ(anchor.sizes(), anchor.grad().sizes());
        ASSERT_EQ(positive.sizes(), positive.grad().sizes());
        ASSERT_EQ(negative.sizes(), negative.grad().sizes());
      }
    }
  }
}

TEST_F(ModulesTest, TripletMarginWithDistanceLossFunctionalParity) {
  // Check for parity between F::triplet_margin_with_distance_loss and
  // TripletMarginWithDistanceLoss.
  auto pairwise_distance = [&](const smith::Tensor& x, const smith::Tensor& y) {
    return smith::pairwise_distance(x, y);
  };
  auto cosine_distance = [&](const smith::Tensor& x, const smith::Tensor& y) {
    return 1.0 - smith::cosine_similarity(x, y);
  };
  std::vector<TripletMarginWithDistanceLossOptions::distance_function_t>
      distance_functions = {pairwise_distance, cosine_distance};

  std::vector<TripletMarginWithDistanceLossOptions::reduction_t> reductions = {
      smith::kSum, smith::kMean, smith::kNone};
  std::vector<float> margins = {0.5, 1.0, 1.5};
  std::vector<bool> swaps = {true, false};

  for (auto& function : distance_functions) {
    for (auto& reduction : reductions) {
      for (auto& margin : margins) {
        for (const auto swap : swaps) {
          auto moduleOptions = TripletMarginWithDistanceLossOptions()
                                   .distance_function(function)
                                   .reduction(reduction)
                                   .margin(margin)
                                   .swap(swap);
          auto functionOptions =
              smith::nn::functional::TripletMarginWithDistanceLossFuncOptions()
                  .distance_function(function)
                  .reduction(reduction)
                  .margin(margin)
                  .swap(swap);

          auto anchor = smith::randn(
              {100, 128}, smith::dtype(smith::kFloat).requires_grad(true));
          auto positive = smith::randn(
              {100, 128}, smith::dtype(smith::kFloat).requires_grad(true));
          auto negative = smith::randn(
              {100, 128}, smith::dtype(smith::kFloat).requires_grad(true));

          TripletMarginWithDistanceLoss distanceLoss(moduleOptions);

          auto moduleOutput = distanceLoss->forward(anchor, positive, negative);
          auto moduleOperatorOutput = distanceLoss(anchor, positive, negative);
          auto functionOutput =
              smith::nn::functional::triplet_margin_with_distance_loss(
                  anchor, positive, negative, functionOptions);

          ASSERT_TRUE(moduleOutput.allclose(functionOutput, 1e-6, 1e-6));
          ASSERT_TRUE(
              moduleOperatorOutput.allclose(functionOutput, 1e-6, 1e-6));
        }
      }
    }
  }
}

TEST_F(ModulesTest, NLLLoss) {
  NLLLoss loss;
  auto input = smith::tensor(
      {{-0.1315, -3.1315, -2.5315},
       {-3.7038, -0.1038, -2.6038},
       {-2.3422, -1.3422, -0.4422}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({1, 0, 2}, smith::kLong);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor(2.4258, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_TRUE(
      NLLLoss(NLLLossOptions().ignore_index(-100).reduction(smith::kMean))
          ->forward(input, target)
          .allclose(expected, 1e-04));
}

TEST_F(ModulesTest, CrossEntropyLoss) {
  CrossEntropyLoss loss;
  auto input = smith::tensor(
      {{3., 3.}, {2., 2.}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({0, 1}, smith::kLong);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor(0.6931, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
  ASSERT_TRUE(
      CrossEntropyLoss(
          CrossEntropyLossOptions().ignore_index(-100).reduction(smith::kMean))
          ->forward(input, target)
          .allclose(expected, 1e-04));

  // label smoothing with class indices
  loss = CrossEntropyLoss(
      CrossEntropyLossOptions().label_smoothing(0.15).reduction(smith::kMean));
  input = smith::tensor(
      {{3., 1.}, {1., 2.}}, smith::dtype(smith::kFloat).requires_grad(true));
  target = smith::tensor({0, 1}, smith::kLong);
  output = loss->forward(input, target);
  expected = smith::tensor(0.3326, smith::kFloat);
  s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input.sizes(), input.grad().sizes());

  // label smoothing with with target probabilities
  loss = CrossEntropyLoss(
      CrossEntropyLossOptions().label_smoothing(0.2).reduction(smith::kMean));
  input = smith::tensor(
      {{3., 1.}, {1., 2.}}, smith::dtype(smith::kFloat).requires_grad(true));
  target = smith::tensor({{0.8, 0.2}, {0.1, 0.9}}, smith::kFloat);
  output = loss->forward(input, target);
  expected = smith::tensor(0.5701, smith::kFloat);
  s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, CosineSimilarity) {
  CosineSimilarity cos(CosineSimilarityOptions().dim(1));
  auto input1 = smith::tensor(
      {{1, 2, 3}, {4, 5, 6}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto input2 = smith::tensor(
      {{1, 8, 3}, {2, 1, 6}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto output = cos->forward(input1, input2);
  auto expected = smith::tensor({0.8078, 0.8721}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected, 1e-04));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}

TEST_F(ModulesTest, SoftMarginLossDefaultOptions) {
  SoftMarginLoss loss;
  auto input = smith::tensor(
      {2., 4., 1., 3.}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({-1., 1., 1., -1.}, smith::kFloat);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({1.3767317}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelSoftMarginLossDefaultOptions) {
  MultiLabelSoftMarginLoss loss;
  auto input = smith::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto target =
      smith::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, smith::kFloat);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({0.7608436}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, SoftMarginLossNoReduction) {
  SoftMarginLoss loss(smith::kNone);
  auto input = smith::tensor(
      {2., 4., 1., 3.}, smith::dtype(smith::kFloat).requires_grad(true));
  auto target = smith::tensor({-1., 1., 1., -1.}, smith::kFloat);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor(
      {2.1269281, 0.01814993, 0.3132617, 3.0485873}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, MultiLabelSoftMarginLossWeightedNoReduction) {
  auto input = smith::tensor(
      {{0., 2., 2., 0.}, {2., 1., 0., 1.}},
      smith::dtype(smith::kFloat).requires_grad(true));
  auto target =
      smith::tensor({{0., 0., 1., 0.}, {1., 0., 1., 1.}}, smith::kFloat);
  auto weight = smith::tensor({0.1, 0.6, 0.4, 0.8}, smith::kFloat);
  auto options =
      MultiLabelSoftMarginLossOptions().reduction(smith::kNone).weight(weight);
  MultiLabelSoftMarginLoss loss = MultiLabelSoftMarginLoss(options);
  auto output = loss->forward(input, target);
  auto expected = smith::tensor({0.4876902, 0.3321295}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input.sizes(), input.grad().sizes());
}

TEST_F(ModulesTest, PairwiseDistance) {
  PairwiseDistance dist(PairwiseDistanceOptions().p(1));
  auto input1 = smith::tensor(
      {{1, 2, 3}, {4, 5, 6}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto input2 = smith::tensor(
      {{1, 8, 3}, {2, 1, 6}}, smith::dtype(smith::kFloat).requires_grad(true));
  auto output = dist->forward(input1, input2);
  auto expected = smith::tensor({6, 6}, smith::kFloat);
  auto s = output.sum();
  s.backward();

  ASSERT_TRUE(output.allclose(expected));
  ASSERT_EQ(input1.sizes(), input1.grad().sizes());
}

TEST_F(ModulesTest, ELU) {
  const auto size = 3;
  for (const auto alpha : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    for (const auto inplace : {false, true}) {
      ELU model{ELUOptions().alpha(alpha).inplace(inplace)};
      auto x = smith::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      if (!inplace) {
        x.requires_grad_(true);
      }
      auto x_orig = x.clone();
      auto y = model(x);
      smith::Tensor s = y.sum();

      ASSERT_EQ(s.ndimension(), 0);

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      auto y_exp = smith::max(smith::zeros_like(x_orig), x_orig) +
          smith::min(smith::zeros_like(x_orig), alpha * (smith::expm1(x_orig)));
      ASSERT_TRUE(smith::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(smith::allclose(x, y_exp));
      } else {
        s.backward();
      }
    }
  }
}

TEST_F(ModulesTest, SELU) {
  for (const auto inplace : {false, true}) {
    SELU model(inplace);
    auto input = smith::randn({5, 5});
    if (!inplace) {
      input.requires_grad_(true);
    }
    auto input_orig = input.clone();
    auto output = model->forward(input);
    const double scale = 1.0507009873554804934193349852946;
    const double alpha = 1.6732632423543772848170429916717;
    auto zero = smith::zeros_like(input);
    auto expected = scale *
        (smith::max(zero, input_orig) +
         smith::min(zero, alpha * (smith::expm1(input_orig))));
    auto s = output.sum();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
    if (inplace) {
      ASSERT_TRUE(input.allclose(expected));
    } else {
      s.backward();
    }
  }
}

TEST_F(ModulesTest, Hardshrink) {
  const auto size = 3;
  for (const auto lambda : {-4.2, -1.0, -0.42, 0.0, 0.42, 1.0, 4.2, 42.42}) {
    Hardshrink model{HardshrinkOptions().lambda(lambda)};
    auto x = smith::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    smith::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x.abs() > lambda) * x;
    ASSERT_TRUE(smith::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, Hardtanh) {
  const auto size = 3;
  for (const auto min_val : {-4.2, -1.0, -0.42, 0.0}) {
    for (const auto max_val : {0.42, 1.0, 4.2}) {
      for (const auto inplace : {false, true}) {
        Hardtanh model{
            HardtanhOptions().min_val(min_val).max_val(max_val).inplace(
                inplace)};
        auto x = smith::linspace(-10.0, 10.0, size * size * size);
        x.resize_({size, size, size});
        if (!inplace) {
          x.requires_grad_(true);
        }
        auto x_orig = x.clone();
        auto y = model(x);
        smith::Tensor s = y.sum();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        auto y_exp = (x_orig < min_val) * min_val +
            ((x_orig >= min_val) * (x_orig <= max_val)) * x_orig +
            (x_orig > max_val) * max_val;
        ASSERT_TRUE(smith::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(smith::allclose(x, y_exp));
        } else {
          s.backward();
        }
      }
    }
  }
}

TEST_F(ModulesTest, HardtanhMinValGEMaxVal) {
  ASSERT_THROWS_WITH(
      Hardtanh{HardtanhOptions().min_val(0.42).max_val(0.42)},
      "max_val must be greater than min_val");
  ASSERT_THROWS_WITH(
      Hardtanh{HardtanhOptions().min_val(0.42).max_val(-0.42)},
      "max_val must be greater than min_val");

  Hardtanh ht{HardtanhOptions().min_val(-0.42).max_val(0.42)};
  ht->options.min_val(0.42);
  ASSERT_THROWS_WITH(ht->reset(), "max_val must be greater than min_val");
  ht->options.max_val(-0.42);
  ASSERT_THROWS_WITH(ht->reset(), "max_val must be greater than min_val");
}

TEST_F(ModulesTest, LeakyReLU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    for (const auto negative_slope : {0.0, 0.42, 1.0}) {
      for (const auto type : {smith::kFloat, smith::kBFloat16}) {
        LeakyReLU model{
            LeakyReLUOptions().negative_slope(negative_slope).inplace(inplace)};
        auto x = smith::linspace(-10.0, 10.0, size * size * size).to(type);
        x.resize_({size, size, size});
        if (!inplace) {
          x.requires_grad_(true);
        }
        auto x_orig = x.clone();
        auto y = model(x);
        smith::Tensor s = y.sum();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        auto y_exp =
            (x_orig < 0) * x_orig * negative_slope + (x_orig >= 0) * x_orig;
        ASSERT_TRUE(smith::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(smith::allclose(x, y_exp));
        } else {
          s.backward();
        }
      }
    }
  }
}

TEST_F(ModulesTest, LogSigmoid) {
  const auto size = 3;
  LogSigmoid model;
  auto x = smith::linspace(-10.0, 10.0, size * size * size);
  x.resize_({size, size, size}).set_requires_grad(true);
  auto y = model(x);
  smith::Tensor s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), 3);
  ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
  auto y_exp = smith::log(
      smith::ones_like(x) / (smith::ones_like(x) + smith::exp(smith::neg(x))));
  ASSERT_TRUE(smith::allclose(y, y_exp, 1e-4, 1e-7));
}

TEST_F(ModulesTest, Softmax) {
  Softmax m(/*dim=*/1);
  auto input = smith::arange(10, smith::kFloat).reshape({2, 5});
  auto output = m(input);
  auto sum = smith::sum(smith::exp(input), 1);

  for (const auto i : c10::irange(2)) {
    auto expected = smith::exp(input[i]) / sum[i];
    ASSERT_TRUE(smith::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, Softmin) {
  Softmin m(/*dim=*/1);
  auto input = smith::arange(10, smith::kFloat).reshape({2, 5});
  auto output = m(input);
  auto sum = smith::sum(smith::exp(-input), 1);

  for (const auto i : c10::irange(2)) {
    auto expected = smith::exp(-input[i]) / sum[i];
    ASSERT_TRUE(smith::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, LogSoftmax) {
  LogSoftmax m(/*dim=*/1);
  auto input = smith::arange(10, smith::kFloat).reshape({2, 5});
  auto output = m(input);
  auto sum = smith::sum(smith::exp(input), 1);

  for (const auto i : c10::irange(2)) {
    auto expected = smith::log(smith::exp(input[i]) / sum[i]);
    ASSERT_TRUE(smith::allclose(output[i], expected));
  }
}

TEST_F(ModulesTest, AdaptiveLogSoftmaxWithLoss) {
  {
    // log_probs actually returns log_proba
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 4, {2}).div_value(2.));
    auto x = smith::randn({4, 8});
    auto logprob_out = asfm->log_prob(x);
    ASSERT_TRUE(
        smith::allclose(smith::exp(logprob_out).data().sum(1), smith::ones(4)));
  }
  {
    // test predict
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 10, {4, 8})
            .div_value(2.)
            .head_bias(true));
    auto x = smith::randn({64, 8});
    auto logprob_out = asfm->log_prob(x);
    auto predict_out = asfm->predict(x);
    ASSERT_TRUE(smith::allclose(predict_out, logprob_out.argmax(1)));
  }
  {
    // cluster sizes
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(2.));
    auto x = smith::arange(100, 132, smith::kFloat).reshape({2, 16});
    auto y = smith::tensor({0, 17}, smith::kLong);
    auto asm_out = asfm(x, y);
    ASSERT_EQ(asm_out.output.sizes(), std::vector<int64_t>({2}));
  }
  {
    // forward returns the same thing as log_probs
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 4, {2}).div_value(2.));
    auto x = smith::randn({4, 8});
    auto logprob_out = asfm->log_prob(x);
    NLLLoss nll_loss;

    for (const auto v : c10::irange(4)) {
      auto y = smith::full({4}, v, smith::kLong);
      auto asm_out = asfm(x, y);
      auto out = asm_out.output;
      auto loss = smith::tensor(asm_out.loss, smith::kFloat);
      auto expected = nll_loss->forward(logprob_out, y);

      ASSERT_TRUE(smith::allclose(loss, expected));
      ASSERT_TRUE(smith::allclose(
          out, logprob_out.gather(1, y.unsqueeze(1)).squeeze()));
    }
  }
  {
    // test no batch dim
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(2.));
    auto x = smith::randn({1, 16});
    auto y = smith::tensor({17});
    auto x2 = x.squeeze(0);
    auto y2 = y.squeeze(0);
    ASSERT_TRUE(
        smith::allclose(asfm(x, y).output.squeeze(0), asfm(x2, y2).output));
  }
  {
    // test div_value
    auto options =
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(0.);
    ASSERT_THROWS_WITH(
        AdaptiveLogSoftmaxWithLoss(options),
        "div_value should not be equal to 0");

    options =
        AdaptiveLogSoftmaxWithLossOptions(16, 20, {4, 10, 15}).div_value(0.25);
    ASSERT_TRUE(AdaptiveLogSoftmaxWithLoss(options));
  }
}

TEST_F(ModulesTest, Softmax2d) {
  Softmax2d m;
  auto input = smith::arange(24, smith::kFloat).reshape({1, 2, 3, 4});
  auto output = m(input);
  auto sum = smith::sum(smith::exp(input), 1);

  for (const auto i : c10::irange(1)) {
    for (const auto j : c10::irange(2)) {
      for (const auto k : c10::irange(3)) {
        for (const auto l : c10::irange(4)) {
          auto expected = smith::exp(input[i][j][k][l]) / sum[i][k][l];
          ASSERT_TRUE(smith::allclose(output[i][j][k][l], expected));
        }
      }
    }
  }
}

TEST_F(ModulesTest, PReLU) {
  const auto num_parameters = 42;
  const auto init = 0.42;

  PReLU model{PReLUOptions().num_parameters(num_parameters).init(init)};

  ASSERT_EQ(model->weight.sizes(), std::vector<int64_t>({num_parameters}));
  ASSERT_TRUE(
      smith::allclose(model->weight, smith::full(num_parameters, init)));

  const auto x = smith::rand({100, num_parameters}) * 200 - 100;
  const auto y = model(x);
  const auto s = y.sum();

  s.backward();
  ASSERT_EQ(s.ndimension(), 0);

  ASSERT_EQ(y.ndimension(), x.ndimension());
  ASSERT_EQ(y.sizes(), x.sizes());
  const auto y_exp = (x < 0) * model->weight * x + (x >= 0) * x;
  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, ReLU) {
  for (const auto inplace : {false, true}) {
    const auto size = 3;
    ReLU model(inplace);
    auto x = smith::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    if (!inplace) {
      x.requires_grad_(true);
    }
    auto x_orig = x.clone();
    auto y = model(x);
    smith::Tensor s = y.sum();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x_orig < 0) * 0 + (x_orig >= 0) * x_orig;
    ASSERT_TRUE(smith::allclose(y, y_exp));
    if (inplace) {
      ASSERT_TRUE(smith::allclose(x, y_exp));
    } else {
      s.backward();
    }
  }
}

TEST_F(ModulesTest, ReLU6) {
  for (const auto inplace : {false, true}) {
    const auto size = 3;
    ReLU6 model(inplace);
    auto x = smith::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size});
    if (!inplace) {
      x.requires_grad_(true);
    }
    auto x_orig = x.clone();
    auto y = model(x);
    smith::Tensor s = y.sum();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x_orig < 0) * 0 + ((x_orig >= 0) * (x_orig <= 6)) * x_orig +
        (x_orig > 6) * 6;
    ASSERT_TRUE(smith::allclose(y, y_exp));
    if (inplace) {
      ASSERT_TRUE(smith::allclose(x, y_exp));
    } else {
      s.backward();
    }
  }
}

TEST_F(ModulesTest, RReLU) {
  const auto size = 3;
  for (const auto lower : {0.01, 0.1, 0.2}) {
    for (const auto upper : {0.3, 0.4, 0.5}) {
      for (const auto inplace : {false, true}) {
        for (const auto type : {smith::kFloat, smith::kBFloat16}) {
          RReLU model{
              RReLUOptions().lower(lower).upper(upper).inplace(inplace)};
          auto x = smith::linspace(-10.0, 10.0, size * size * size).to(type);
          x.resize_({size, size, size});
          if (!inplace) {
            x.requires_grad_(true);
          }
          auto x_orig = x.clone();
          auto y = model(x);
          smith::Tensor s = y.sum();

          ASSERT_EQ(s.ndimension(), 0);
          ASSERT_EQ(y.ndimension(), 3);
          ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
          auto z =
              ((x_orig >= 0) * (x_orig == y) +
               (x_orig < 0) * (y >= x_orig * upper) * (y <= lower * x_orig)) *
              1.0;
          ASSERT_TRUE(smith::allclose(z, smith::ones_like(z)));
          if (inplace) {
            ASSERT_TRUE(smith::allclose(x, y));
          } else {
            s.backward();
          }
        }
      }
    }
  }
}

TEST_F(ModulesTest, CELU) {
  const auto size = 3;
  for (const auto inplace : {false, true}) {
    for (const auto alpha : {0.42, 1.0, 4.2, 42.42}) {
      CELU model{CELUOptions().alpha(alpha).inplace(inplace)};
      auto x = smith::linspace(-10.0, 10.0, size * size * size);
      x.resize_({size, size, size});
      if (!inplace) {
        x.requires_grad_(true);
      }
      auto x_orig = x.clone();
      auto y = model(x);
      smith::Tensor s = y.sum();

      ASSERT_EQ(s.ndimension(), 0);
      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      auto y_exp = smith::max(smith::zeros_like(x_orig), x_orig) +
          smith::min(smith::zeros_like(x_orig),
                     alpha * (smith::expm1(x_orig / alpha)));
      ASSERT_TRUE(smith::allclose(y, y_exp));
      if (inplace) {
        ASSERT_TRUE(smith::allclose(x, y_exp));
      } else {
        s.backward();
      }
    }
  }
}

TEST_F(ModulesTest, GLU) {
  int64_t dim = 1;
  GLU model(dim);
  auto input = smith::randn({4, 2}, smith::requires_grad());
  auto output = model->forward(input);
  auto input_size = input.sizes()[dim] / 2;
  auto first_half = input.narrow(dim, 0, input_size);
  auto second_half = input.narrow(dim, input_size, input_size);
  auto expected = first_half * smith::sigmoid(second_half);
  auto s = output.sum();
  s.backward();

  ASSERT_EQ(s.ndimension(), 0);
  ASSERT_TRUE(output.allclose(expected));

  GLU model_default_options;
  ASSERT_TRUE(model_default_options->forward(input).allclose(expected));
}

TEST_F(ModulesTest, GELU) {
  GELU model(GELUOptions().approximate("none"));
  const auto x = smith::linspace(-3.0, 3.0, 100);
  const auto y_exp = x * 0.5 * (1.0 + smith::erf(x / std::sqrt(2.0)));
  const auto y = model(x);
  ASSERT_TRUE(smith::allclose(y, y_exp, 1.4e-06, 1e-05));
}

TEST_F(ModulesTest, TanhGELU) {
  GELU model(GELUOptions().approximate("tanh"));
  const auto x = smith::linspace(-3.0, 3.0, 100);
  const auto inner = std::sqrt(2 / M_PI) * (x + 0.044715 * x.pow(3.0));
  const auto y_exp = 0.5 * x * (1.0 + inner.tanh());
  const auto y = model(x);
  ASSERT_TRUE(smith::allclose(y, y_exp, 1.4e-06, 1e-05));
}

TEST_F(ModulesTest, Mish) {
  Mish model;
  auto x = smith::randn(100) * 10;
  auto y_exp = x * x.exp().log1p().tanh();
  auto y = model(x);

  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, Sigmoid) {
  Sigmoid model;
  auto x = smith::randn(100) * 10;
  auto y_exp = 1 / (1 + smith::exp(-x));
  auto y = model(x);

  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, PixelShuffle) {
  PixelShuffle module(/*upscale_factor=*/2);
  auto x = smith::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      smith::kFloat);
  auto y_exp = smith::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      smith::kFloat);
  auto y = module(x);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(y.sizes(), smith::IntArrayRef({1, 1, 4, 4}));
  ASSERT_TRUE(y.allclose(y_exp));
}

TEST_F(ModulesTest, PixelUnshuffle) {
  PixelUnshuffle module(/*downscale_factor=*/2);
  auto x = smith::tensor(
      {{{{-17, 7, 19, 14}, {0, -15, -2, 0}, {-1, -3, 2, 1}, {-12, -3, 14, 9}}}},
      smith::kFloat);
  auto y_exp = smith::tensor(
      {{{{-17, 19}, {-1, 2}},
        {{7, 14}, {-3, 1}},
        {{0, -2}, {-12, 14}},
        {{-15, 0}, {-3, 9}}}},
      smith::kFloat);
  auto y = module(x);

  ASSERT_EQ(y.ndimension(), 4);
  ASSERT_EQ(y.sizes(), smith::IntArrayRef({1, 4, 2, 2}));
  ASSERT_TRUE(y.allclose(y_exp));
}

TEST_F(ModulesTest, Softplus) {
  const auto size = 3;
  for (const auto beta : {0.5, 1.0, 2.0}) {
    for (const auto threshold : {1.0, 3.0, 5.0}) {
      Softplus model{SoftplusOptions().beta(beta).threshold(threshold)};
      auto x = smith::linspace(-3.0, 3.0, 61);
      x.resize_({size, size, size});
      auto y_exp =
          (x <= threshold) * smith::log(1 + smith::exp(x * beta)) / beta +
          (x > threshold) * x;
      auto y = model(x);

      ASSERT_EQ(y.ndimension(), 3);
      ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
      ASSERT_TRUE(smith::allclose(y, y_exp));
    }
  }
}

TEST_F(ModulesTest, Softshrink) {
  const auto size = 3;
  for (const auto lambda : {0.0, 0.42, 1.0, 4.2, 42.42}) {
    Softshrink model{/*lambda=*/lambda};
    auto x = smith::linspace(-10.0, 10.0, size * size * size);
    x.resize_({size, size, size}).set_requires_grad(true);
    auto y = model(x);
    smith::Tensor s = y.sum();

    s.backward();
    ASSERT_EQ(s.ndimension(), 0);

    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
    auto y_exp = (x < -lambda) * (x + lambda) + (x > lambda) * (x - lambda);
    ASSERT_TRUE(smith::allclose(y, y_exp));
  }
}

TEST_F(ModulesTest, Softsign) {
  Softsign model;
  auto x = smith::randn(100) * 10;
  auto y_exp = x / (1 + x.abs());
  auto y = model(x);

  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, Tanh) {
  Tanh model;
  auto x = smith::randn(100) * 10;
  auto y_exp = (x.exp() - (-x).exp()) / (x.exp() + (-x).exp());
  auto y = model(x);

  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, Tanhshrink) {
  Tanhshrink model;
  auto x = smith::randn(100) * 10;
  auto y_exp = x - x.tanh();
  auto y = model(x);

  ASSERT_TRUE(smith::allclose(y, y_exp));
}

TEST_F(ModulesTest, Threshold) {
  const auto size = 3;
  for (const auto threshold : {0.5, 1.0, 2.0}) {
    for (const auto value : {0.5, 1.0, 2.0}) {
      for (const auto inplace : {false, true}) {
        Threshold model{ThresholdOptions(threshold, value).inplace(inplace)};
        auto x = smith::linspace(-3.0, 3.0, 61);
        x.resize_({size, size, size});
        auto x_orig = x.clone();
        auto y_exp =
            (x_orig <= threshold) * value + (x_orig > threshold) * x_orig;
        auto y = model(x);

        ASSERT_EQ(y.ndimension(), 3);
        ASSERT_EQ(y.sizes(), std::vector<int64_t>({size, size, size}));
        ASSERT_TRUE(smith::allclose(y, y_exp));
        if (inplace) {
          ASSERT_TRUE(smith::allclose(x, y_exp));
        }
      }
    }
  }
}

TEST_F(ModulesTest, Upsampling1D) {
  {
    Upsample model(UpsampleOptions()
                       .size(std::vector<int64_t>({4}))
                       .mode(smith::kNearest));
    auto input = smith::ones({1, 1, 2}, smith::requires_grad());
    auto output = model->forward(input);
    auto expected = smith::ones({1, 1, 4});
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(UpsampleOptions()
                           .scale_factor(std::vector<double>({scale_factor}))
                           .mode(smith::kLinear)
                           .align_corners(align_corners));
        auto input = smith::ones({1, 1, 2}, smith::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = smith::ones({1, 1, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    // linear (1D) upsampling spatial invariance
    Upsample model(UpsampleOptions()
                       .scale_factor(std::vector<double>({3}))
                       .mode(smith::kLinear)
                       .align_corners(false));
    auto input = smith::zeros({1, 1, 9});
    input.narrow(2, 0, 4).normal_();
    auto output = model->forward(input);
    auto expected = model->forward(input.narrow(2, 0, 5));

    ASSERT_TRUE(smith::allclose(output.narrow(2, 0, 15), expected));
  }
}

TEST_F(ModulesTest, Upsampling2D) {
  {
    Upsample model(UpsampleOptions()
                       .size(std::vector<int64_t>({4, 4}))
                       .mode(smith::kNearest));
    auto input = smith::ones({1, 1, 2, 2}, smith::requires_grad());
    auto output = model->forward(input);
    auto expected = smith::ones({1, 1, 4, 4});
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(
            UpsampleOptions()
                .scale_factor(std::vector<double>({scale_factor, scale_factor}))
                .mode(smith::kBilinear)
                .align_corners(align_corners));
        auto input = smith::ones({1, 1, 2, 2}, smith::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = smith::ones({1, 1, expected_size, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(
            UpsampleOptions()
                .scale_factor(std::vector<double>({scale_factor, scale_factor}))
                .mode(smith::kBicubic)
                .align_corners(align_corners));
        auto input = smith::ones({1, 1, 2, 2}, smith::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected = smith::ones({1, 1, expected_size, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
}

TEST_F(ModulesTest, Upsampling3D) {
  {
    Upsample model(UpsampleOptions()
                       .size(std::vector<int64_t>({4, 4, 4}))
                       .mode(smith::kNearest));
    auto input = smith::ones({1, 1, 2, 2, 2}, smith::requires_grad());
    auto output = model->forward(input);
    auto expected = smith::ones({1, 1, 4, 4, 4});
    auto s = output.sum();
    s.backward();

    ASSERT_EQ(s.ndimension(), 0);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    for (const auto align_corners : {true, false}) {
      // test float scale factor up & down sampling
      for (const auto scale_factor : {0.5, 1.5, 2.0}) {
        Upsample model(UpsampleOptions()
                           .scale_factor(std::vector<double>(
                               {scale_factor, scale_factor, scale_factor}))
                           .mode(smith::kTrilinear)
                           .align_corners(align_corners));
        auto input = smith::ones({1, 1, 2, 2, 2}, smith::requires_grad());
        auto output = model->forward(input);
        auto expected_size =
            static_cast<int64_t>(std::floor(input.size(-1) * scale_factor));
        auto expected =
            smith::ones({1, 1, expected_size, expected_size, expected_size});
        auto s = output.sum();
        s.backward();

        ASSERT_EQ(s.ndimension(), 0);
        ASSERT_TRUE(output.allclose(expected));
      }
    }
  }
}

TEST_F(ModulesTest, CTCLoss) {
  CTCLoss loss{CTCLossOptions().reduction(smith::kNone)};
  const auto target_lengths = smith::tensor({0, 0, 0});
  const auto input_lengths = smith::tensor({50, 50, 50});
  const auto targets =
      smith::randint(1, 15, at::IntArrayRef({0}), smith::kLong);
  const auto log_probs =
      smith::randn({50, 3, 15}, smith::kDouble).log_softmax(2);
  const auto output =
      loss->forward(log_probs, targets, input_lengths, target_lengths);
  ASSERT_TRUE(output.ge(0).all().item<bool>());
  ASSERT_TRUE(smith::allclose(
      -log_probs.sum(0).slice(1, 0, 1).view_as(output), output));
}

TEST_F(ModulesTest, PoissonNLLLoss) {
  const auto input = smith::tensor({0.5, 1.5, 2.5});
  const auto target = smith::tensor({1., 2., 3.});
  const auto component_wise_loss = smith::exp(input) - target * input;
  {
    PoissonNLLLoss loss{PoissonNLLLossOptions().reduction(smith::kNone)};
    ASSERT_TRUE(
        smith::allclose(component_wise_loss, loss->forward(input, target)));
  }
  {
    PoissonNLLLoss loss{PoissonNLLLossOptions().reduction(smith::kSum)};
    ASSERT_TRUE(smith::allclose(
        smith::sum(component_wise_loss), loss->forward(input, target)));
  }
  {
    PoissonNLLLoss loss{PoissonNLLLossOptions().reduction(smith::kMean)};
    ASSERT_TRUE(smith::allclose(
        smith::mean(component_wise_loss), loss->forward(input, target)));
  }
}

TEST_F(ModulesTest, MarginRankingLoss) {
  {
    MarginRankingLoss loss;
    const auto input1 = smith::randn(15) * 10;
    const auto input2 = smith::randn(15) * 10;
    const auto target = smith::randn(15).sign();
    ASSERT_TRUE(smith::allclose(
        loss->forward(input1, input2, target),
        (-target * (input1 - input2)).clamp(0).mean()));
  }
  {
    MarginRankingLoss loss{
        MarginRankingLossOptions().margin(0.5).reduction(smith::kSum)};
    const auto input1 = smith::randn(15) * 10;
    const auto input2 = smith::randn(15) * 10;
    const auto target = smith::randn(15).sign();
    const auto margin = 0.5;
    ASSERT_TRUE(smith::allclose(
        loss->forward(input1, input2, target),
        (-target * (input1 - input2) + margin).clamp(0).sum()));
  }
  {
    MarginRankingLoss loss{
        MarginRankingLossOptions().margin(0.5).reduction(smith::kMean)};
    const auto input1 = smith::randn(15) * 10;
    const auto input2 = smith::randn(15) * 10;
    const auto target = smith::randn(15).sign();
    const auto margin = 0.5;
    ASSERT_TRUE(smith::allclose(
        loss->forward(input1, input2, target),
        (-target * (input1 - input2) + margin).clamp(0).mean()));
  }
}

TEST_F(ModulesTest, BCEWithLogitsLoss) {
  { // test BCE with logits raises if target and input are different size
    {
      const auto target = smith::rand(5);
      const auto input = smith::rand({5, 1});
      ASSERT_THROWS_WITH(
          BCEWithLogitsLoss()(input, target), "must be the same as input size");
    }
    {
      const auto target = smith::rand({5, 1});
      const auto input = smith::rand(5);
      ASSERT_THROWS_WITH(
          BCEWithLogitsLoss()(input, target), "must be the same as input size");
    }
  }
  { // test BCE with logits gives same result as sigmoid and bce loss
    auto sigmoid = Sigmoid();

    auto target = smith::rand({64, 4});
    auto output = smith::rand({64, 4}) - 0.5;

    ASSERT_TRUE(smith::allclose(
        BCEWithLogitsLoss()(output, target),
        BCELoss()(sigmoid(output), target)));

    auto weight = smith::rand(4);
    ASSERT_TRUE(smith::allclose(
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
            output, target),
        BCELoss(BCELossOptions().weight(weight))(sigmoid(output), target)));

    target = smith::zeros({4, 1}, smith::kFloat);
    output = smith::empty({4, 1}, smith::kFloat).fill_(-100);

    ASSERT_TRUE(smith::allclose(
        BCEWithLogitsLoss()(output, target),
        BCELoss()(sigmoid(output), target)));

    ASSERT_TRUE(smith::allclose(
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().reduction(smith::kNone))(
            output, target),
        BCELoss(BCELossOptions().reduction(smith::kNone))(
            sigmoid(output), target)));

    weight = smith::rand({1}, smith::kFloat);
    ASSERT_TRUE(smith::allclose(
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
            output, target),
        BCELoss(BCELossOptions().weight(weight))(sigmoid(output), target)));
  }
  { // test BCE with logits has correct grad at zero
    const auto output = smith::zeros({3, 1}, smith::requires_grad());
    const auto target = smith::zeros({3, 1});
    BCEWithLogitsLoss(BCEWithLogitsLossOptions().reduction(smith::kSum))(
        output, target)
        .backward();
    const auto expected_grad = smith::empty({3, 1}).fill_(0.5);
    ASSERT_TRUE(smith::allclose(output.grad(), expected_grad));
  }
  { // test BCE with logits broadcasts weights
    const auto target = smith::rand({16, 4});
    const auto output = smith::rand({16, 4}) - 0.5;

    auto weight = smith::rand(4);
    auto out1 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    weight = weight.expand({16, 4}).contiguous();
    auto out2 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    ASSERT_TRUE(smith::allclose(out1, out2));

    weight = smith::rand({16, 1});
    out1 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    weight = weight.expand({16, 4}).contiguous();
    out2 = BCEWithLogitsLoss(BCEWithLogitsLossOptions().weight(weight))(
        output, target);

    ASSERT_TRUE(smith::allclose(out1, out2));
  }
  { // test BCE with logits ones in pos weights are the same as none
    const auto target = smith::rand({64, 4});
    const auto output = smith::rand({64, 4}) - 0.5;
    const auto pos_weight = smith::ones({64, 4});

    ASSERT_TRUE(smith::allclose(
        BCEWithLogitsLoss()(output, target),
        BCEWithLogitsLoss(BCEWithLogitsLossOptions().pos_weight(pos_weight))(
            output, target)));
  }
  { // test BCE with logits broadcasts pos weights
    const auto target = smith::rand({64, 4});
    const auto output = smith::rand({64, 4}) - 0.5;
    const auto pos_weight = smith::rand(4);
    const auto out1 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);

    const auto pos_weight1 = pos_weight.expand({1, 4});
    const auto out2 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);

    const auto pos_weight2 = pos_weight.expand({64, 4});
    const auto out3 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);

    ASSERT_TRUE(smith::allclose(out1, out2));
    ASSERT_TRUE(smith::allclose(out1, out3));
  }
  { // test BCE with logits with pos weight has correct grad at zero
    const auto output = smith::zeros({3, 1}, smith::requires_grad());
    const auto target = smith::zeros({3, 1});
    const auto pos_weight = smith::ones({3, 1});
    BCEWithLogitsLoss(BCEWithLogitsLossOptions()
                          .pos_weight(pos_weight)
                          .reduction(smith::kSum))(output, target)
        .backward();
    const auto expected_grad = smith::empty({3, 1}).fill_(0.5);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const auto grad = output.grad();
    ASSERT_TRUE(smith::allclose(grad, expected_grad));
  }
  { // test BCE with logits stability
    const auto output = smith::tensor({0., -120.});
    const auto target = smith::tensor({0., 1.});
    const auto pos_weight = smith::tensor({1., 1.});

    const auto out1 = BCEWithLogitsLoss()(output, target);
    ASSERT_TRUE(smith::isfinite(out1).all().item<bool>());

    const auto out2 = BCEWithLogitsLoss(
        BCEWithLogitsLossOptions().pos_weight(pos_weight))(output, target);
    ASSERT_TRUE(smith::isfinite(out2).all().item<bool>());
  }
}

namespace detail {

namespace F = smith::nn::functional;

smith::Tensor _batchmatmul(const smith::Tensor& a, const smith::Tensor& b) {
  SMITH_INTERNAL_ASSERT(a.size(0) == b.size(0));
  SMITH_INTERNAL_ASSERT(a.size(1) == b.size(1));
  auto retval = smith::zeros(
      {a.size(0), a.size(1), a.size(2), b.size(3)}, smith::kFloat32);
  for (const auto i : c10::irange(a.size(0))) {
    for (const auto j : c10::irange(a.size(1))) {
      retval[i][j] = smith::matmul(a[i][j], b[i][j]);
    }
  }
  return retval;
}

smith::Tensor _softmax(const smith::Tensor& x) {
  auto output = smith::zeros(x.sizes());
  for (const auto i : c10::irange(x.size(0))) {
    for (const auto j : c10::irange(x.size(1))) {
      for (const auto k : c10::irange(x.size(2))) {
        const auto& x_curr = x[i][j][k];
        const auto e_x = smith::exp(x_curr - smith::max(x_curr));
        output[i][j][k] = e_x / smith::sum(e_x);
      }
    }
  }
  return output;
}

std::tuple<smith::Tensor, smith::Tensor> _scaled_dot_attn_ref(
    const smith::Tensor& Q,
    const smith::Tensor& K,
    const smith::Tensor& V,
    at::IntArrayRef dims,
    const smith::Tensor& unseen_mask = {},
    const smith::Tensor& key_padding_mask = {},
    bool average_attn_weights = true) {
  auto QKT = _batchmatmul(Q, K.permute({0, 1, 3, 2}) / std::sqrt(dims[3]));
  const auto b1 = QKT.size(0);
  const auto b2 = QKT.size(1);
  const auto s1 = QKT.size(2);
  const auto s2 = QKT.size(3);
  if (unseen_mask.defined() || key_padding_mask.defined()) {
    for (const auto i : c10::irange(b1)) {
      for (const auto j : c10::irange(b2)) {
        for (const auto m : c10::irange(s1)) {
          for (const auto n : c10::irange(s2)) {
            if (unseen_mask.defined() &&
                unseen_mask[m][n].item<double>() == 0) {
              QKT[i][j][m][n] = -std::numeric_limits<double>::infinity();
            }
            if (key_padding_mask.defined() &&
                key_padding_mask[i][n].item<double>() != 0) {
              QKT[i][j][m][n] = -std::numeric_limits<double>::infinity();
            }
          }
        }
      }
    }
  }
  auto reference = _softmax(QKT);
  auto ref_attn_weight = reference;
  if (average_attn_weights) {
    // NOLINTNEXTLINE(bugprone-argument-comment)
    ref_attn_weight = smith::sum(ref_attn_weight, /*axis=*/1) / b2;
  }
  reference = _batchmatmul(reference, V);
  return std::tie(reference, ref_attn_weight);
}

smith::Tensor _split_heads_ref(
    const smith::Tensor& X,
    at::IntArrayRef dims,
    int nheads,
    int d_head) {
  auto X_split = X.reshape({dims[0], dims[1], nheads, d_head});
  auto X_split_transposed = X_split.permute({0, 2, 1, 3});
  return X_split_transposed.reshape({dims[0], nheads, dims[1], d_head});
}

smith::Tensor _combine_heads_ref(
    const smith::Tensor& X,
    at::IntArrayRef dims,
    int nheads,
    int d_head) {
  auto X_transposed = X.permute({0, 2, 1, 3});
  auto reference = X_transposed.reshape({dims[0], dims[1], nheads * d_head});
  return reference;
}

smith::Tensor _fc(
    smith::Tensor X,
    smith::Tensor X_weight,
    smith::Tensor X_bias) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto X_fc_b = X_bias;
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto X_fc_w = X_weight;
  return smith::matmul(X, smith::t(X_fc_w)) + X_fc_b;
}

void _multihead_attn_test_helper(
    bool add_key_padding_mask = false,
    bool add_bias_kv = false,
    bool add_zero_attn = false,
    bool saved_kv = false,
    bool same_embed_dim = false,
    bool average_attn_weights = true) {
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<int> d_2_10(2, 10);
  std::uniform_int_distribution<int> d_3_10(3, 10);
  bool registration_checked = false;
  for ([[maybe_unused]] const auto i : c10::irange(100)) {
    const auto batch_sz = d_2_10(generator);
    const auto seq_len = d_2_10(generator);
    const auto d_head = d_3_10(generator);
    const auto nheads = d_3_10(generator);
    const auto d_model = d_head * nheads;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int kv_dim;
    if (same_embed_dim) {
      kv_dim = d_model;
    } else {
      std::uniform_int_distribution<int> d(5, 20);
      kv_dim = d(generator);
      while (kv_dim == d_model) {
        kv_dim = d(generator);
      }
    }
    std::vector<int64_t> dims{batch_sz, seq_len, kv_dim};
    smith::Tensor saved_k;
    smith::Tensor saved_k_tensor;
    smith::Tensor saved_v;
    smith::Tensor saved_v_tensor;
    if (saved_kv) {
      saved_k = smith::rand({batch_sz * nheads, seq_len, d_head});
      saved_k_tensor = saved_k;
      saved_v = smith::rand({batch_sz * nheads, seq_len, d_head});
      saved_v_tensor = saved_v;
    }
    smith::Tensor key_padding_mask;
    smith::Tensor key_padding_mask_tensor;
    if (add_key_padding_mask) {
      const auto seq_mask = smith::randint(0, 2, {1, seq_len});
      key_padding_mask = seq_mask.repeat({batch_sz, 1}) == 1;
      key_padding_mask_tensor = key_padding_mask;
    }
    const auto decoder_state = smith::rand({batch_sz, d_model});
    const smith::Tensor K = smith::rand(dims);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const smith::Tensor V = K;
    const smith::Tensor Q =
        decoder_state.clone().resize_({batch_sz, 1, d_model});
    auto attn_mask = smith::randint(0, 2, {1, seq_len}, smith::kFloat);
    const smith::Tensor attn_mask_tensor = attn_mask.clone();
    attn_mask_tensor.masked_fill_(
        attn_mask_tensor == 0, -std::numeric_limits<double>::infinity());
    attn_mask_tensor.masked_fill_(attn_mask_tensor > 0, double(0.0));

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const smith::Tensor decoder_state_tensor = decoder_state;
    const smith::Tensor source_hid_tensor = K.transpose(0, 1);

    const auto options = MultiheadAttentionOptions(d_model, nheads)
                             .add_bias_kv(add_bias_kv)
                             .add_zero_attn(add_zero_attn)
                             .kdim(kv_dim)
                             .vdim(kv_dim);
    const auto multihead_attn_module = MultiheadAttention(options);

    if (!registration_checked) {
      // make sure parameters are all registered correctly
      auto named_parameters = multihead_attn_module->named_parameters();
      if (same_embed_dim) {
        ASSERT_TRUE(named_parameters.contains("in_proj_weight"));
      } else {
        ASSERT_TRUE(named_parameters.contains("q_proj_weight"));
        ASSERT_TRUE(named_parameters.contains("k_proj_weight"));
        ASSERT_TRUE(named_parameters.contains("v_proj_weight"));
      }
      if (add_bias_kv) {
        ASSERT_TRUE(named_parameters.contains("bias_k"));
        ASSERT_TRUE(named_parameters.contains("bias_v"));
      }
      // make sure sub modules are all registered correctly
      auto submodules = multihead_attn_module->named_children();
      ASSERT_TRUE(submodules.contains("out_proj"));
      registration_checked = true;
    }

    smith::Tensor bias_k;
    smith::Tensor bias_v;
    if (add_bias_kv) {
      bias_k = multihead_attn_module->bias_k.detach();
      bias_v = multihead_attn_module->bias_v.detach();
    } else {
      bias_k.reset();
      bias_v.reset();
    }

    smith::Tensor _Q = decoder_state_tensor.unsqueeze(1).transpose(0, 1);
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    smith::Tensor _V = source_hid_tensor;
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    smith::Tensor _K = source_hid_tensor;

    smith::Tensor result;
    smith::Tensor result_weight;
    if (multihead_attn_module->_qkv_same_embed_dim) {
      std::tie(result, result_weight) = F::multi_head_attention_forward(
          _Q,
          _K,
          _V,
          F::MultiheadAttentionForwardFuncOptions(
              /*embed_dim_to_check=*/d_model,
              /*num_heads=*/nheads,
              /*in_proj_weight=*/multihead_attn_module->in_proj_weight,
              /*in_proj_bias=*/multihead_attn_module->in_proj_bias,
              /*bias_k=*/multihead_attn_module->bias_k,
              /*bias_v=*/multihead_attn_module->bias_v,
              /*add_zero_attn=*/multihead_attn_module->options.add_zero_attn(),
              /*dropout_p=*/multihead_attn_module->options.dropout(),
              /*out_proj_weight=*/multihead_attn_module->out_proj->weight,
              /*out_proj_bias=*/multihead_attn_module->out_proj->bias)
              .training(multihead_attn_module->is_training())
              .key_padding_mask(key_padding_mask_tensor)
              .need_weights(true)
              .attn_mask(attn_mask_tensor)
              .static_k(saved_k_tensor)
              .static_v(saved_v_tensor)
              .average_attn_weights(average_attn_weights));
    } else {
      std::tie(result, result_weight) = F::multi_head_attention_forward(
          _Q,
          _K,
          _V,
          F::MultiheadAttentionForwardFuncOptions(
              /*embed_dim_to_check=*/d_model,
              /*num_heads=*/nheads,
              /*in_proj_weight=*/{},
              /*in_proj_bias=*/multihead_attn_module->in_proj_bias,
              /*bias_k=*/multihead_attn_module->bias_k,
              /*bias_v=*/multihead_attn_module->bias_v,
              /*add_zero_attn=*/multihead_attn_module->options.add_zero_attn(),
              /*dropout_p=*/multihead_attn_module->options.dropout(),
              /*out_proj_weight=*/multihead_attn_module->out_proj->weight,
              /*out_proj_bias=*/multihead_attn_module->out_proj->bias)
              .training(multihead_attn_module->is_training())
              .key_padding_mask(key_padding_mask_tensor)
              .need_weights(true)
              .attn_mask(attn_mask_tensor)
              .use_separate_proj_weight(true)
              .q_proj_weight(multihead_attn_module->q_proj_weight)
              .k_proj_weight(multihead_attn_module->k_proj_weight)
              .v_proj_weight(multihead_attn_module->v_proj_weight)
              .static_k(saved_k_tensor)
              .static_v(saved_v_tensor)
              .average_attn_weights(average_attn_weights));
    }
    result = result.squeeze(0).detach();
    smith::Tensor q_proj_weight;
    smith::Tensor k_proj_weight;
    smith::Tensor v_proj_weight;
    if (multihead_attn_module->_qkv_same_embed_dim) {
      q_proj_weight =
          multihead_attn_module->in_proj_weight.slice(/*dim=*/0, 0, d_model);
      k_proj_weight = multihead_attn_module->in_proj_weight.slice(
          /*dim=*/0, d_model, (d_model * 2));
      v_proj_weight =
          multihead_attn_module->in_proj_weight.slice(/*dim=*/0, (d_model * 2));
    } else {
      q_proj_weight = multihead_attn_module->q_proj_weight;
      k_proj_weight = multihead_attn_module->k_proj_weight;
      v_proj_weight = multihead_attn_module->v_proj_weight;
    }
    auto Q_fc =
        _fc(Q,
            q_proj_weight,
            multihead_attn_module->in_proj_bias.slice(/*dim=*/0, 0, d_model));
    auto K_fc =
        _fc(K,
            k_proj_weight,
            multihead_attn_module->in_proj_bias.slice(
                /*dim=*/0, d_model, (d_model * 2)));
    auto V_fc = _fc(
        V,
        v_proj_weight,
        multihead_attn_module->in_proj_bias.slice(/*dim=*/0, (d_model * 2)));

    if (add_bias_kv) {
      K_fc = smith::cat(
          {K_fc,
           bias_k.repeat({K_fc.size(0) / bias_k.size(0), 1, 1} /*, axis=0*/)},
          /*dim=*/1);
      V_fc = smith::cat(
          {V_fc,
           bias_v.repeat({V_fc.size(0) / bias_v.size(0), 1, 1} /*, axis=0*/)},
          /*dim=*/1);
      if (attn_mask.defined()) {
        attn_mask = smith::cat({attn_mask, smith::ones({1, 1})}, /*dim=*/1);
      }
      if (key_padding_mask.defined()) {
        key_padding_mask = smith::cat(
            {key_padding_mask, smith::full({batch_sz, 1}, false, smith::kBool)},
            /*dim=*/1);
      }
      dims[1] += 1;
    }
    const auto Q_split =
        _split_heads_ref(Q_fc, {batch_sz, 1, d_model}, nheads, d_head);
    smith::Tensor K_split;
    if (saved_k.defined()) {
      K_split = saved_k.reshape({dims[0], nheads, dims[1], d_head});
    } else {
      K_split = _split_heads_ref(K_fc, dims, nheads, d_head);
    }
    smith::Tensor V_split;
    if (saved_v.defined()) {
      V_split = saved_v.reshape({dims[0], nheads, dims[1], d_head});
    } else {
      V_split = _split_heads_ref(V_fc, dims, nheads, d_head);
    }
    if (add_zero_attn) {
      dims[1] += 1;
      K_split = smith::cat(
          {K_split,
           smith::zeros(
               {K_split.size(0), K_split.size(1), 1, K_split.size(3)})},
          /*dim=*/2);
      V_split = smith::cat(
          {V_split,
           smith::zeros(
               {V_split.size(0), V_split.size(1), 1, V_split.size(3)})},
          /*dim=*/2);
      if (attn_mask.defined()) {
        attn_mask = smith::cat({attn_mask, smith::ones({1, 1})}, /*dim=*/1);
      }
      if (key_padding_mask.defined()) {
        key_padding_mask = smith::cat(
            {key_padding_mask, smith::full({batch_sz, 1}, false, smith::kBool)},
            /*dim=*/1);
      }
    }
    auto [attn_heads, ref_attn_weight] = _scaled_dot_attn_ref(
        Q_split,
        K_split,
        V_split,
        Q_split.sizes(),
        attn_mask,
        key_padding_mask,
        average_attn_weights);
    const auto combined_attn_heads =
        _combine_heads_ref(attn_heads, {batch_sz, 1}, nheads, d_head);
    auto reference =
        _fc(combined_attn_heads,
            multihead_attn_module->out_proj->weight,
            multihead_attn_module->out_proj->bias);
    // NOLINTNEXTLINE(bugprone-argument-comment)
    reference = smith::squeeze(reference, /*axis=*/1);

    // result = reference
    ASSERT_EQ(result.sizes(), std::vector<int64_t>({batch_sz, d_model}));
    ASSERT_TRUE(
        smith::allclose(result, reference, 1e-5, 1e-5, /*equal_nan=*/true));

    // result_weight = ref_attn_weight
    result_weight = result_weight.detach();
    ASSERT_EQ(result_weight.sizes(), ref_attn_weight.sizes());
    ASSERT_TRUE(smith::allclose(
        result_weight, ref_attn_weight, 1e-5, 1e-5, /*equal_nan=*/true));
  }
}
} // namespace detail

TEST_F(ModulesTest, MultiheadAttention) {
  using namespace ::detail;

  for (auto average_attn_weights : {false, true}) {
    // test_multihead_attn_add_zero_attn
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/false,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/true,
        /*saved_kv=*/false,
        /*same_embed_dim=*/false,
        /*average_attn_weights=*/average_attn_weights);

    // test_multihead_attn_add_bias_kv
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/false,
        /*add_bias_kv=*/true,
        /*add_zero_attn=*/false,
        /*saved_kv=*/false,
        /*same_embed_dim=*/false,
        /*average_attn_weights=*/average_attn_weights);

    // test_multihead_attn_no_masking():
    _multihead_attn_test_helper();

    // test_multihead_attn_key_padding_mask
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
        /*saved_kv=*/false,
        /*same_embed_dim=*/false,
        /*average_attn_weights=*/average_attn_weights);

    // test_multihead_attn_saved_kv
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/false,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
        /*saved_kv=*/true,
        /*same_embed_dim=*/false,
        /*average_attn_weights=*/average_attn_weights);

    // test_multihead_attn_add_bias_kv_zero_attn
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,
        /*add_bias_kv=*/true,
        /*add_zero_attn=*/true,
        /*saved_kv=*/false,
        /*same_embed_dim=*/false,
        /*average_attn_weights=*/average_attn_weights);

    // test_multihead_attn_all_arguments1
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/true,
        /*saved_kv=*/true,
        /*same_embed_dim=*/false,
        /*average_attn_weights=*/average_attn_weights);

    ASSERT_THROWS_WITH(
        // test_multihead_attn_all_arguments2
        _multihead_attn_test_helper(
            /*add_key_padding_mask=*/true,
            /*add_bias_kv=*/true,
            /*add_zero_attn=*/true,
            /*saved_kv=*/true,
            /*same_embed_dim=*/false,
            /*average_attn_weights=*/average_attn_weights),
        "bias cannot be added to static key");

    // test_multihead_attn_all_arguments3
    _multihead_attn_test_helper(
        /*add_key_padding_mask=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/true,
        /*saved_kv=*/true,
        /*same_embed_dim=*/true,
        /*average_attn_weights=*/average_attn_weights);
  }
}

TEST_F(ModulesTest, PrettyPrintIdentity) {
  ASSERT_EQ(c10::str(Identity()), "smith::nn::Identity()");
}

TEST_F(ModulesTest, PrettyPrintFlatten) {
  ASSERT_EQ(c10::str(Flatten()), "smith::nn::Flatten(start_dim=1, end_dim=-1)");
  ASSERT_EQ(
      c10::str(Flatten(FlattenOptions().start_dim(2).end_dim(4))),
      "smith::nn::Flatten(start_dim=2, end_dim=4)");
}

TEST_F(ModulesTest, PrettyPrintUnflatten) {
  ASSERT_EQ(
      c10::str(Unflatten(UnflattenOptions(0, {2, 2}))),
      "smith::nn::Unflatten(dim=0, unflattened_size={2, 2})");
  ASSERT_EQ(
      c10::str(Unflatten(UnflattenOptions(
          "B",
          {std::pair<std::string, int64_t>{"B1", 2},
           std::pair<std::string, int64_t>{"B2", 2}}))),
      "smith::nn::Unflatten(dim=\"B\", unflattened_size={{\"B1\", 2}, {\"B2\", 2}})");
}

TEST_F(ModulesTest, ReflectionPad1d) {
  {
    ReflectionPad1d m(ReflectionPad1dOptions(2));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{2., 1., 0., 1., 2., 3., 2., 1.}, {6., 5., 4., 5., 6., 7., 6., 5.}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReflectionPad1d m(ReflectionPad1dOptions({3, 1}));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{3., 2., 1., 0., 1., 2., 3., 2.}, {7., 6., 5., 4., 5., 6., 7., 6.}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReflectionPad2d) {
  {
    ReflectionPad2d m(ReflectionPad2dOptions(2));
    auto input = smith::arange(9, smith::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{8., 7., 6., 7., 8., 7., 6.},
           {5., 4., 3., 4., 5., 4., 3.},
           {2., 1., 0., 1., 2., 1., 0.},
           {5., 4., 3., 4., 5., 4., 3.},
           {8., 7., 6., 7., 8., 7., 6.},
           {5., 4., 3., 4., 5., 4., 3.},
           {2., 1., 0., 1., 2., 1., 0.}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReflectionPad2d m(ReflectionPad2dOptions({1, 1, 2, 0}));
    auto input = smith::arange(9, smith::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{7., 6., 7., 8., 7.},
           {4., 3., 4., 5., 4.},
           {1., 0., 1., 2., 1.},
           {4., 3., 4., 5., 4.},
           {7., 6., 7., 8., 7.}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReflectionPad3d) {
  {
    ReflectionPad3d m(ReflectionPad3dOptions(1));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{7., 6., 7., 6.},
            {5., 4., 5., 4.},
            {7., 6., 7., 6.},
            {5., 4., 5., 4.}},
           {{3., 2., 3., 2.},
            {1., 0., 1., 0.},
            {3., 2., 3., 2.},
            {1., 0., 1., 0.}},
           {{7., 6., 7., 6.},
            {5., 4., 5., 4.},
            {7., 6., 7., 6.},
            {5., 4., 5., 4.}},
           {{3., 2., 3., 2.},
            {1., 0., 1., 0.},
            {3., 2., 3., 2.},
            {1., 0., 1., 0.}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReflectionPad3d m(ReflectionPad3dOptions({0, 1, 1, 0, 1, 2}));
    auto input = smith::arange(16, smith::kFloat).reshape({1, 1, 4, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{6., 7., 6.}, {4., 5., 4.}, {6., 7., 6.}},
           {{2., 3., 2.}, {0., 1., 0.}, {2., 3., 2.}},
           {{6., 7., 6.}, {4., 5., 4.}, {6., 7., 6.}},
           {{10., 11., 10.}, {8., 9., 8.}, {10., 11., 10.}},
           {{14., 15., 14.}, {12., 13., 12.}, {14., 15., 14.}},
           {{10., 11., 10.}, {8., 9., 8.}, {10., 11., 10.}},
           {{6., 7., 6.}, {4., 5., 4.}, {6., 7., 6.}}}}},
        smith::kFloat);
    ASSERT_EQ(output.sizes(), std::vector<int64_t>({1, 1, 7, 3, 3}));
    ASSERT_TRUE(output.allclose(expected));
  }
}
TEST_F(ModulesTest, ReplicationPad1d) {
  {
    ReplicationPad1d m(ReplicationPad1dOptions(2));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{0., 0., 0., 1., 2., 3., 3., 3.}, {4., 4., 4., 5., 6., 7., 7., 7.}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReplicationPad1d m(ReplicationPad1dOptions({3, 1}));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{0., 0., 0., 0., 1., 2., 3., 3.}, {4., 4., 4., 4., 5., 6., 7., 7.}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad2d) {
  {
    ReplicationPad2d m(ReplicationPad2dOptions(2));
    auto input = smith::arange(9, smith::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{0., 0., 0., 1., 2., 2., 2.},
           {0., 0., 0., 1., 2., 2., 2.},
           {0., 0., 0., 1., 2., 2., 2.},
           {3., 3., 3., 4., 5., 5., 5.},
           {6., 6., 6., 7., 8., 8., 8.},
           {6., 6., 6., 7., 8., 8., 8.},
           {6., 6., 6., 7., 8., 8., 8.}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReplicationPad2d m(ReplicationPad2dOptions({1, 1, 2, 0}));
    auto input = smith::arange(9, smith::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{0., 0., 1., 2., 2.},
           {0., 0., 1., 2., 2.},
           {0., 0., 1., 2., 2.},
           {3., 3., 4., 5., 5.},
           {6., 6., 7., 8., 8.}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ReplicationPad3d) {
  {
    ReplicationPad3d m(ReplicationPad3dOptions(1));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{0., 0., 1., 1.},
            {0., 0., 1., 1.},
            {2., 2., 3., 3.},
            {2., 2., 3., 3.}},
           {{0., 0., 1., 1.},
            {0., 0., 1., 1.},
            {2., 2., 3., 3.},
            {2., 2., 3., 3.}},
           {{4., 4., 5., 5.},
            {4., 4., 5., 5.},
            {6., 6., 7., 7.},
            {6., 6., 7., 7.}},
           {{4., 4., 5., 5.},
            {4., 4., 5., 5.},
            {6., 6., 7., 7.},
            {6., 6., 7., 7.}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ReplicationPad3d m(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{0., 0., 1., 1., 1.},
            {0., 0., 1., 1., 1.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.}},
           {{0., 0., 1., 1., 1.},
            {0., 0., 1., 1., 1.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.},
            {2., 2., 3., 3., 3.}},
           {{4., 4., 5., 5., 5.},
            {4., 4., 5., 5., 5.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.}},
           {{4., 4., 5., 5., 5.},
            {4., 4., 5., 5., 5.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.}},
           {{4., 4., 5., 5., 5.},
            {4., 4., 5., 5., 5.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.},
            {6., 6., 7., 7., 7.}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ZeroPad1d) {
  {
    ZeroPad1d m(ZeroPad1dOptions(2));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{0., 0., 0., 1., 2., 3., 0., 0.}, {0., 0., 4., 5., 6., 7., 0., 0.}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ZeroPad1d m(ZeroPad1dOptions({3, 1}));
    auto input = smith::arange(6, smith::kFloat).reshape({1, 2, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{0., 0., 0., 0., 1., 2., 0.}, {0., 0., 0., 3., 4., 5., 0.}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ZeroPad2d) {
  {
    ZeroPad2d m(ZeroPad2dOptions(2));
    auto input = smith::arange(9, smith::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{0., 0., 0., 0., 0., 0., 0.},
           {0., 0., 0., 0., 0., 0., 0.},
           {0., 0., 0., 1., 2., 0., 0.},
           {0., 0., 3., 4., 5., 0., 0.},
           {0., 0., 6., 7., 8., 0., 0.},
           {0., 0., 0., 0., 0., 0., 0.},
           {0., 0., 0., 0., 0., 0., 0.}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ZeroPad2d m(ZeroPad2dOptions({1, 1, 2, 0}));
    auto input = smith::arange(9, smith::kFloat).reshape({1, 1, 3, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{0., 0., 0., 0., 0.},
           {0., 0., 0., 0., 0.},
           {0., 0., 1., 2., 0.},
           {0., 3., 4., 5., 0.},
           {0., 6., 7., 8., 0.}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ZeroPad3d) {
  {
    ZeroPad3d m(ZeroPad3dOptions(1));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{0., 0., 0., 0.},
            {0., 0., 0., 0.},
            {0., 0., 0., 0.},
            {0., 0., 0., 0.}},
           {{0., 0., 0., 0.},
            {0., 0., 1., 0.},
            {0., 2., 3., 0.},
            {0., 0., 0., 0.}},
           {{0., 0., 0., 0.},
            {0., 4., 5., 0.},
            {0., 6., 7., 0.},
            {0., 0., 0., 0.}},
           {{0., 0., 0., 0.},
            {0., 0., 0., 0.},
            {0., 0., 0., 0.},
            {0., 0., 0., 0.}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ZeroPad3d m(ZeroPad3dOptions({1, 2, 1, 2, 1, 2}));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},
            {0., 0., 1., 0., 0.},
            {0., 2., 3., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},
            {0., 4., 5., 0., 0.},
            {0., 6., 7., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}},
           {{0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0.}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad1d) {
  {
    ConstantPad1d m(ConstantPad1dOptions(2, 3.5));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 2, 4});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{3.5000, 3.5000, 0.0000, 1.0000, 2.0000, 3.0000, 3.5000, 3.5000},
          {3.5000, 3.5000, 4.0000, 5.0000, 6.0000, 7.0000, 3.5000, 3.5000}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ConstantPad1d m(ConstantPad1dOptions({3, 1}, 3.5));
    auto input = smith::arange(6, smith::kFloat).reshape({1, 2, 3});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{3.5000, 3.5000, 3.5000, 0.0000, 1.0000, 2.0000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.0000, 4.0000, 5.0000, 3.5000}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad2d) {
  {
    ConstantPad2d m(ConstantPad2dOptions(2, 3.5));
    auto input = smith::arange(4, smith::kFloat).reshape({1, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 0.0000, 1.0000, 3.5000, 3.5000},
          {3.5000, 3.5000, 2.0000, 3.0000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ConstantPad2d m(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
    auto input = smith::arange(4, smith::kFloat).reshape({1, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
          {3.5000, 3.5000, 3.5000, 0.0000, 1.0000},
          {3.5000, 3.5000, 3.5000, 2.0000, 3.0000},
          {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, ConstantPad3d) {
  {
    ConstantPad3d m(ConstantPad3dOptions(1, 3.5));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 0.0000, 1.0000, 3.5000},
            {3.5000, 2.0000, 3.0000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 4.0000, 5.0000, 3.5000},
            {3.5000, 6.0000, 7.0000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
  {
    ConstantPad3d m(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
    auto input = smith::arange(8, smith::kFloat).reshape({1, 1, 2, 2, 2});
    auto output = m(input);
    auto expected = smith::tensor(
        {{{{{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 0.0000, 1.0000, 3.5000, 3.5000},
            {3.5000, 2.0000, 3.0000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 4.0000, 5.0000, 3.5000, 3.5000},
            {3.5000, 6.0000, 7.0000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}},
           {{3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000},
            {3.5000, 3.5000, 3.5000, 3.5000, 3.5000}}}}},
        smith::kFloat);
    ASSERT_TRUE(output.allclose(expected));
  }
}

TEST_F(ModulesTest, CrossMapLRN2d) {
  /// size 3, default options
  auto input =
      smith::arange(9, smith::kFloat32).view({1, 1, 3, 3}).requires_grad_(true);
  auto expected = smith::tensor(
      {{{{0.00000000, 0.99997497, 1.99980010},
         {2.99932500, 3.99840070, 4.99687700},
         {5.99460600, 6.99143740, 7.98722360}}}},
      smith::kFloat32);
  auto grad_expected = smith::tensor(
      {{{{1.00000000, 0.99992496, 0.99970007},
         {0.99932520, 0.99880093, 0.99812720},
         {0.99730474, 0.99633380, 0.99521490}}}},
      smith::kFloat32);
  auto crossmaplrn2d = CrossMapLRN2d(3);
  auto output = crossmaplrn2d(input);
  output.sum().backward();

  ASSERT_TRUE(input.grad().allclose(grad_expected));
  ASSERT_TRUE(output.allclose(expected));

  /// size change
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(4).alpha(1e-4).beta(0.75).k(1));
  output = crossmaplrn2d(input);
  expected = smith::tensor(
      {{{{0.00000000, 0.99998120, 1.99985000},
         {2.99949400, 3.99880050, 4.99765800},
         {5.99595300, 6.99357600, 7.99041300}}}},
      smith::kFloat32);
  ASSERT_TRUE(output.allclose(expected));

  /// alpha change
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-3).beta(0.75).k(1));
  output = crossmaplrn2d(input);
  expected = smith::tensor(
      {{{{0.00000000, 0.99975010, 1.99800230},
         {2.99326750, 3.98407440, 4.96897600},
         {5.94656100, 6.91545720, 7.87434340}}}},
      smith::kFloat32);
  ASSERT_TRUE(output.allclose(expected));

  /// beta change
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-4).beta(0.95).k(1));
  output = crossmaplrn2d(input);
  expected = smith::tensor(
      {{{{0.00000000, 0.99996830, 1.99974680},
         {2.99914500, 3.99797440, 4.99604460},
         {5.99316840, 6.98915600, 7.98382000}}}},
      smith::kFloat32);
  ASSERT_TRUE(output.allclose(expected));

  /// k change
  crossmaplrn2d =
      CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-4).beta(0.75).k(2));
  output = crossmaplrn2d(input);
  expected = smith::tensor(
      {{{{0.00000000, 0.59459610, 1.18914770},
         {1.78361000, 2.37793870, 2.97208900},
         {3.56601700, 4.15967700, 4.75302650}}}},
      smith::kFloat32);
  ASSERT_TRUE(output.allclose(expected));
}

TEST_F(ModulesTest, RNNCell) {
  smith::manual_seed(0);
  auto rnn = RNNCell(1, 2);

  auto input = smith::randn({3, 1});
  auto hx = smith::randn({3, 2});
  auto output = rnn(input, hx);
  auto expected =
      smith::tensor({{-0.5078, 0.4380}, {-0.7215, 0.2969}, {-0.1304, 0.0653}});
  ASSERT_TRUE(smith::allclose(output, expected, 1e-05, 2e-04));

  output = rnn(input);
  expected =
      smith::tensor({{-0.0775, 0.6688}, {-0.0734, 0.4759}, {-0.0725, 0.4225}});
  ASSERT_TRUE(smith::allclose(output, expected, 1e-05, 2e-04));

  input = smith::randn({1});
  hx = smith::randn({2});
  output = rnn(input, hx);
  expected = smith::tensor({0.2808, 0.6505});
  ASSERT_TRUE(smith::allclose(output, expected, 1e-05, 2e-04));

  {
    auto input = smith::randn({3, 2});
    auto hx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        rnn(input, hx), "input has inconsistent input_size: got 2 expected 1");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 1});
    ASSERT_THROWS_WITH(
        rnn(input, hx),
        "hidden0 has inconsistent hidden_size: got 1, expected 2");
  }

  {
    auto input = smith::randn({3, 1, 1, 1, 1});
    auto hx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        rnn(input, hx), "Expected input to be 1D or 2D, got 5D instead");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 1, 1, 1, 2});
    ASSERT_THROWS_WITH(
        rnn(input, hx), "Expected hidden to be 1D or 2D, got 5D instead");
  }
}

TEST_F(ModulesTest, LSTMCell) {
  smith::manual_seed(0);
  auto lstm = LSTMCell(1, 2);

  auto input = smith::randn({3, 1});
  auto hx = smith::randn({3, 2});
  auto cx = smith::randn({3, 2});
  auto output = lstm(input, std::make_tuple(hx, cx));
  auto output_hx = std::get<0>(output);
  auto output_cx = std::get<1>(output);
  auto expected_hx =
      smith::tensor({{-0.2462, 0.0810}, {-0.2206, 0.1867}, {-0.0146, 0.0429}});
  auto expected_cx =
      smith::tensor({{-0.4480, 0.1071}, {-0.6245, 0.2687}, {-0.0322, 0.0518}});
  ASSERT_TRUE(smith::allclose(output_hx, expected_hx, 1e-05, 2e-04));
  ASSERT_TRUE(smith::allclose(output_cx, expected_cx, 1e-05, 2e-04));

  output = lstm(input);
  output_hx = std::get<0>(output);
  output_cx = std::get<1>(output);
  expected_hx =
      smith::tensor({{-0.1331, 0.1634}, {-0.1494, 0.2869}, {-0.1428, 0.2263}});
  expected_cx =
      smith::tensor({{-0.2679, 0.2180}, {-0.3049, 0.3493}, {-0.2896, 0.2853}});
  ASSERT_TRUE(smith::allclose(output_hx, expected_hx, 1e-05, 2e-04));
  ASSERT_TRUE(smith::allclose(output_cx, expected_cx, 1e-05, 2e-04));

  input = smith::randn({1});
  hx = smith::randn({2});
  cx = smith::randn({2});
  output = lstm(input, std::make_tuple(hx, cx));
  output_hx = std::get<0>(output);
  output_cx = std::get<1>(output);
  expected_hx = smith::tensor({-0.0443, 0.1537});
  expected_cx = smith::tensor({-0.1195, 0.2144});
  ASSERT_TRUE(smith::allclose(output_hx, expected_hx, 1e-05, 2e-04));
  ASSERT_TRUE(smith::allclose(output_cx, expected_cx, 1e-05, 2e-04));

  {
    auto input = smith::randn({3, 2});
    auto hx = smith::randn({3, 2});
    auto cx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "input has inconsistent input_size: got 2 expected 1");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 1});
    auto cx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "hidden0 has inconsistent hidden_size: got 1, expected 2");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 2});
    auto cx = smith::randn({3, 1});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "hidden1 has inconsistent hidden_size: got 1, expected 2");
  }

  {
    auto input = smith::randn({3, 1, 1, 1, 1});
    auto hx = smith::randn({3, 1});
    auto cx = smith::randn({3, 1});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "Expected input to be 1D or 2D, got 5D instead");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 1, 1, 1, 2});
    auto cx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "Expected hx[0] to be 1D or 2D, got 5D instead");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 2});
    auto cx = smith::randn({3, 1, 1, 1, 2});
    ASSERT_THROWS_WITH(
        lstm(input, std::make_tuple(hx, cx)),
        "Expected hx[1] to be 1D or 2D, got 5D instead");
  }
}

TEST_F(ModulesTest, GRUCell) {
  smith::manual_seed(0);
  auto gru = GRUCell(1, 2);

  auto input = smith::randn({3, 1});
  auto hx = smith::randn({3, 2});
  auto output = gru(input, hx);
  auto expected =
      smith::tensor({{1.0243, 0.3227}, {-0.5659, 0.0330}, {-0.4030, -0.2800}});
  ASSERT_TRUE(smith::allclose(output, expected, 1e-05, 2e-04));

  output = gru(input);
  expected =
      smith::tensor({{-0.0085, 0.1095}, {-0.1291, 0.2675}, {-0.1339, 0.2725}});
  ASSERT_TRUE(smith::allclose(output, expected, 1e-05, 2e-04));

  input = smith::randn({1});
  hx = smith::randn({2});
  output = gru(input, hx);
  expected = smith::tensor({-1.0058, -0.3025});
  ASSERT_TRUE(smith::allclose(output, expected, 1e-05, 2e-04));

  {
    auto input = smith::randn({3, 2});
    auto hx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        gru(input, hx), "input has inconsistent input_size: got 2 expected 1");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 1});
    ASSERT_THROWS_WITH(
        gru(input, hx),
        "hidden0 has inconsistent hidden_size: got 1, expected 2");
  }

  {
    auto input = smith::randn({3, 1, 1, 1, 1});
    auto hx = smith::randn({3, 2});
    ASSERT_THROWS_WITH(
        gru(input, hx), "Expected input to be 1D or 2D, got 5D instead");
  }

  {
    auto input = smith::randn({3, 1});
    auto hx = smith::randn({3, 1, 1, 1, 2});
    ASSERT_THROWS_WITH(
        gru(input, hx), "Expected hidden to be 1D or 2D, got 5D instead");
  }
}

TEST_F(ModulesTest, PrettyPrintLinear) {
  ASSERT_EQ(
      c10::str(Linear(3, 4)),
      "smith::nn::Linear(in_features=3, out_features=4, bias=true)");
}

TEST_F(ModulesTest, PrettyPrintBilinear) {
  ASSERT_EQ(
      c10::str(Bilinear(3, 2, 4)),
      "smith::nn::Bilinear(in1_features=3, in2_features=2, out_features=4, bias=true)");
  ASSERT_EQ(
      c10::str(Bilinear(BilinearOptions(3, 2, 4).bias(false))),
      "smith::nn::Bilinear(in1_features=3, in2_features=2, out_features=4, bias=false)");
}

TEST_F(ModulesTest, PrettyPrintConv) {
  ASSERT_EQ(
      c10::str(Conv1d(3, 4, 5)),
      "smith::nn::Conv1d(3, 4, kernel_size=5, stride=1)");
  {
    auto options = Conv1dOptions(3, 4, 5);
    ASSERT_EQ(
        c10::str(Conv1d(options.padding(smith::kSame))),
        "smith::nn::Conv1d(3, 4, kernel_size=5, stride=1, padding='same')");
    ASSERT_EQ(
        c10::str(Conv1d(options.padding(smith::kValid))),
        "smith::nn::Conv1d(3, 4, kernel_size=5, stride=1, padding='valid')");
  }

  ASSERT_EQ(
      c10::str(Conv2d(3, 4, 5)),
      "smith::nn::Conv2d(3, 4, kernel_size=[5, 5], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(Conv2d(Conv2dOptions(3, 4, 5).stride(2))),
      "smith::nn::Conv2d(3, 4, kernel_size=[5, 5], stride=[2, 2])");
  {
    const auto options =
        Conv2dOptions(3, 4, std::vector<int64_t>{5, 6}).stride({1, 2});
    ASSERT_EQ(
        c10::str(Conv2d(options)),
        "smith::nn::Conv2d(3, 4, kernel_size=[5, 6], stride=[1, 2])");
  }
  {
    auto options = Conv2dOptions(3, 4, std::vector<int64_t>{5, 6});
    ASSERT_EQ(
        c10::str(Conv2d(options.padding(smith::kSame))),
        "smith::nn::Conv2d(3, 4, kernel_size=[5, 6], stride=[1, 1], padding='same')");
    ASSERT_EQ(
        c10::str(Conv2d(options.padding(smith::kValid))),
        "smith::nn::Conv2d(3, 4, kernel_size=[5, 6], stride=[1, 1], padding='valid')");
  }

  ASSERT_EQ(
      c10::str(Conv3d(4, 4, std::vector<int64_t>{5, 6, 7})),
      "smith::nn::Conv3d(4, 4, kernel_size=[5, 6, 7], stride=[1, 1, 1])");
  {
    const auto options = Conv3dOptions(4, 4, std::vector<int64_t>{5, 6, 7})
                             .stride({1, 2, 3})
                             .padding(1)
                             .dilation(0)
                             .groups(2)
                             .bias(false)
                             .padding_mode(smith::kCircular);
    ASSERT_EQ(
        c10::str(Conv3d(options)),
        "smith::nn::Conv3d("
        "4, "
        "4, "
        "kernel_size=[5, 6, 7], "
        "stride=[1, 2, 3], "
        "padding=[1, 1, 1], "
        "dilation=[0, 0, 0], "
        "groups=2, "
        "bias=false, "
        "padding_mode=kCircular)");
  }
  {
    auto options = Conv3dOptions(3, 4, std::vector<int64_t>{5, 6, 7});
    ASSERT_EQ(
        c10::str(Conv3d(options.padding(smith::kSame))),
        "smith::nn::Conv3d(3, 4, kernel_size=[5, 6, 7], stride=[1, 1, 1], padding='same')");
    ASSERT_EQ(
        c10::str(Conv3d(options.padding(smith::kValid))),
        "smith::nn::Conv3d(3, 4, kernel_size=[5, 6, 7], stride=[1, 1, 1], padding='valid')");
  }
}

TEST_F(ModulesTest, PrettyPrintConvTranspose) {
  ASSERT_EQ(
      c10::str(ConvTranspose1d(3, 4, 5)),
      "smith::nn::ConvTranspose1d(3, 4, kernel_size=5, stride=1)");

  ASSERT_EQ(
      c10::str(ConvTranspose2d(3, 4, 5)),
      "smith::nn::ConvTranspose2d(3, 4, kernel_size=[5, 5], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(ConvTranspose2d(ConvTranspose2dOptions(3, 4, 5).stride(2))),
      "smith::nn::ConvTranspose2d(3, 4, kernel_size=[5, 5], stride=[2, 2])");
  {
    const auto options =
        ConvTranspose2dOptions(3, 4, std::vector<int64_t>{5, 6}).stride({1, 2});
    ASSERT_EQ(
        c10::str(ConvTranspose2d(options)),
        "smith::nn::ConvTranspose2d(3, 4, kernel_size=[5, 6], stride=[1, 2])");
  }

  ASSERT_EQ(
      c10::str(ConvTranspose3d(4, 4, std::vector<int64_t>{5, 6, 7})),
      "smith::nn::ConvTranspose3d(4, 4, kernel_size=[5, 6, 7], stride=[1, 1, 1])");
  {
    const auto options =
        ConvTranspose3dOptions(4, 4, std::vector<int64_t>{5, 6, 7})
            .stride({1, 2, 3})
            .padding(1)
            .dilation(0)
            .groups(2)
            .bias(false)
            .padding_mode(smith::kCircular);
    ASSERT_EQ(
        c10::str(ConvTranspose3d(options)),
        "smith::nn::ConvTranspose3d("
        "4, "
        "4, "
        "kernel_size=[5, 6, 7], "
        "stride=[1, 2, 3], "
        "padding=[1, 1, 1], "
        "dilation=[0, 0, 0], "
        "groups=2, "
        "bias=false, "
        "padding_mode=kCircular)");
  }
}

TEST_F(ModulesTest, PrettyPrintUpsample) {
  ASSERT_EQ(
      c10::str(
          Upsample(UpsampleOptions().size(std::vector<int64_t>({2, 4, 4})))),
      "smith::nn::Upsample(size=[2, 4, 4], mode=kNearest)");
  ASSERT_EQ(
      c10::str(Upsample(UpsampleOptions()
                            .scale_factor(std::vector<double>({0.5, 1.5}))
                            .mode(smith::kBilinear))),
      "smith::nn::Upsample(scale_factor=[0.5, 1.5], mode=kBilinear)");
}

TEST_F(ModulesTest, PrettyPrintFold) {
  ASSERT_EQ(
      c10::str(Fold(FoldOptions({2, 2}, {5, 5}))),
      "smith::nn::Fold(output_size=[2, 2], kernel_size=[5, 5], dilation=[1, 1], padding=[0, 0], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(Fold(
          FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2, 1}).stride(2))),
      "smith::nn::Fold(output_size=[8, 8], kernel_size=[3, 3], dilation=[2, 2], padding=[2, 1], stride=[2, 2])");
}

TEST_F(ModulesTest, PrettyPrintUnfold) {
  ASSERT_EQ(
      c10::str(Unfold(smith::IntArrayRef({2, 4}))),
      "smith::nn::Unfold(kernel_size=[2, 4], dilation=[1, 1], padding=[0, 0], stride=[1, 1])");
  ASSERT_EQ(
      c10::str(
          Unfold(UnfoldOptions({2, 4}).dilation(2).padding({2, 1}).stride(2))),
      "smith::nn::Unfold(kernel_size=[2, 4], dilation=[2, 2], padding=[2, 1], stride=[2, 2])");
}

TEST_F(ModulesTest, PrettyPrintMaxPool) {
  ASSERT_EQ(
      c10::str(MaxPool1d(5)),
      "smith::nn::MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool2d(5)),
      "smith::nn::MaxPool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool2d(MaxPool2dOptions(5).stride(2))),
      "smith::nn::MaxPool2d(kernel_size=[5, 5], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool3d(5)),
      "smith::nn::MaxPool3d(kernel_size=[5, 5, 5], stride=[5, 5, 5], padding=[0, 0, 0], dilation=[1, 1, 1], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(MaxPool3d(MaxPool3dOptions(5).stride(2))),
      "smith::nn::MaxPool3d(kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=[0, 0, 0], dilation=[1, 1, 1], ceil_mode=false)");

  const auto options =
      MaxPool2dOptions(std::vector<int64_t>{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(MaxPool2d(options)),
      "smith::nn::MaxPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=false)");
}

TEST_F(ModulesTest, PrettyPrintAvgPool) {
  ASSERT_EQ(
      c10::str(AvgPool1d(5)),
      "smith::nn::AvgPool1d(kernel_size=5, stride=5, padding=0)");
  ASSERT_EQ(
      c10::str(AvgPool2d(5)),
      "smith::nn::AvgPool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(AvgPool2d(AvgPool2dOptions(5).stride(2))),
      "smith::nn::AvgPool2d(kernel_size=[5, 5], stride=[2, 2], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(AvgPool3d(5)),
      "smith::nn::AvgPool3d(kernel_size=[5, 5, 5], stride=[5, 5, 5], padding=[0, 0, 0])");
  ASSERT_EQ(
      c10::str(AvgPool3d(AvgPool3dOptions(5).stride(2))),
      "smith::nn::AvgPool3d(kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=[0, 0, 0])");

  const auto options =
      AvgPool2dOptions(std::vector<int64_t>{5, 6}).stride({1, 2});
  ASSERT_EQ(
      c10::str(AvgPool2d(options)),
      "smith::nn::AvgPool2d(kernel_size=[5, 6], stride=[1, 2], padding=[0, 0])");
}

TEST_F(ModulesTest, PrettyPrinFractionalMaxPool) {
  ASSERT_EQ(
      c10::str(
          FractionalMaxPool2d(FractionalMaxPool2dOptions(5).output_size(1))),
      "smith::nn::FractionalMaxPool2d()");
  ASSERT_EQ(
      c10::str(
          FractionalMaxPool3d(FractionalMaxPool3dOptions(5).output_size(1))),
      "smith::nn::FractionalMaxPool3d()");
}

TEST_F(ModulesTest, PrettyPrintLPPool) {
  ASSERT_EQ(
      c10::str(LPPool1d(2, 5)),
      "smith::nn::LPPool1d(norm_type=2, kernel_size=5, stride=5, ceil_mode=false)");
  ASSERT_EQ(
      c10::str(LPPool1d(LPPool1dOptions(1, 2).stride(5).ceil_mode(true))),
      "smith::nn::LPPool1d(norm_type=1, kernel_size=2, stride=5, ceil_mode=true)");
  ASSERT_EQ(
      c10::str(LPPool2d(2, std::vector<int64_t>({1, 2}))),
      "smith::nn::LPPool2d(norm_type=2, kernel_size=[1, 2], stride=[1, 2], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(LPPool2d(LPPool2dOptions(1, std::vector<int64_t>({3, 4}))
                            .stride({5, 6})
                            .ceil_mode(true))),
      "smith::nn::LPPool2d(norm_type=1, kernel_size=[3, 4], stride=[5, 6], ceil_mode=true)");
  ASSERT_EQ(
      c10::str(LPPool3d(2, std::vector<int64_t>({1, 2, 3}))),
      "smith::nn::LPPool3d(norm_type=2, kernel_size=[1, 2, 3], stride=[1, 2, 3], ceil_mode=false)");
  ASSERT_EQ(
      c10::str(LPPool3d(LPPool3dOptions(1, std::vector<int64_t>({3, 4, 5}))
                            .stride({5, 6, 7})
                            .ceil_mode(true))),
      "smith::nn::LPPool3d(norm_type=1, kernel_size=[3, 4, 5], stride=[5, 6, 7], ceil_mode=true)");
}

TEST_F(ModulesTest, PrettyPrintAdaptiveMaxPool) {
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool1d(5)),
      "smith::nn::AdaptiveMaxPool1d(output_size=5)");

  const auto options = AdaptiveMaxPool1dOptions(3);
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool1d(options)),
      "smith::nn::AdaptiveMaxPool1d(output_size=3)");

  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(5)),
      "smith::nn::AdaptiveMaxPool2d(output_size=[5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(AdaptiveMaxPool2dOptions({5, 6}))),
      "smith::nn::AdaptiveMaxPool2d(output_size=[5, 6])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(AdaptiveMaxPool2dOptions({5, std::nullopt}))),
      "smith::nn::AdaptiveMaxPool2d(output_size=[5, None])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool2d(
          AdaptiveMaxPool2dOptions({std::nullopt, std::nullopt}))),
      "smith::nn::AdaptiveMaxPool2d(output_size=[None, None])");

  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(5)),
      "smith::nn::AdaptiveMaxPool3d(output_size=[5, 5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(AdaptiveMaxPool3dOptions({5, 6, 7}))),
      "smith::nn::AdaptiveMaxPool3d(output_size=[5, 6, 7])");
  ASSERT_EQ(
      c10::str(
          AdaptiveMaxPool3d(AdaptiveMaxPool3dOptions({5, std::nullopt, 7}))),
      "smith::nn::AdaptiveMaxPool3d(output_size=[5, None, 7])");
  ASSERT_EQ(
      c10::str(AdaptiveMaxPool3d(AdaptiveMaxPool3dOptions(
          {std::nullopt, std::nullopt, std::nullopt}))),
      "smith::nn::AdaptiveMaxPool3d(output_size=[None, None, None])");
}

TEST_F(ModulesTest, PrettyPrintAdaptiveAvgPool) {
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool1d(5)),
      "smith::nn::AdaptiveAvgPool1d(output_size=5)");

  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(5)),
      "smith::nn::AdaptiveAvgPool2d(output_size=[5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({5, 6}))),
      "smith::nn::AdaptiveAvgPool2d(output_size=[5, 6])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({5, std::nullopt}))),
      "smith::nn::AdaptiveAvgPool2d(output_size=[5, None])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool2d(
          AdaptiveAvgPool2dOptions({std::nullopt, std::nullopt}))),
      "smith::nn::AdaptiveAvgPool2d(output_size=[None, None])");

  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(5)),
      "smith::nn::AdaptiveAvgPool3d(output_size=[5, 5, 5])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions({5, 6, 7}))),
      "smith::nn::AdaptiveAvgPool3d(output_size=[5, 6, 7])");
  ASSERT_EQ(
      c10::str(
          AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions({5, std::nullopt, 7}))),
      "smith::nn::AdaptiveAvgPool3d(output_size=[5, None, 7])");
  ASSERT_EQ(
      c10::str(AdaptiveAvgPool3d(AdaptiveAvgPool3dOptions(
          {std::nullopt, std::nullopt, std::nullopt}))),
      "smith::nn::AdaptiveAvgPool3d(output_size=[None, None, None])");
}

TEST_F(ModulesTest, PrettyPrintMaxUnpool) {
  ASSERT_EQ(
      c10::str(MaxUnpool1d(5)),
      "smith::nn::MaxUnpool1d(kernel_size=5, stride=5, padding=0)");
  ASSERT_EQ(
      c10::str(MaxUnpool1d(MaxUnpool1dOptions(5).stride(3).padding(1))),
      "smith::nn::MaxUnpool1d(kernel_size=5, stride=3, padding=1)");

  ASSERT_EQ(
      c10::str(MaxUnpool2d(5)),
      "smith::nn::MaxUnpool2d(kernel_size=[5, 5], stride=[5, 5], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(MaxUnpool2d(std::vector<int64_t>{5, 6})),
      "smith::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[5, 6], padding=[0, 0])");
  ASSERT_EQ(
      c10::str(MaxUnpool2d(MaxUnpool2dOptions(std::vector<int64_t>{5, 6})
                               .stride({3, 4})
                               .padding({1, 2}))),
      "smith::nn::MaxUnpool2d(kernel_size=[5, 6], stride=[3, 4], padding=[1, 2])");
}

TEST_F(ModulesTest, PrettyPrintDropout) {
  ASSERT_EQ(c10::str(Dropout()), "smith::nn::Dropout(p=0.5, inplace=false)");
  ASSERT_EQ(
      c10::str(Dropout(0.42)), "smith::nn::Dropout(p=0.42, inplace=false)");
  ASSERT_EQ(
      c10::str(Dropout(DropoutOptions().p(0.42).inplace(true))),
      "smith::nn::Dropout(p=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintDropout2d) {
  ASSERT_EQ(
      c10::str(Dropout2d()), "smith::nn::Dropout2d(p=0.5, inplace=false)");
  ASSERT_EQ(
      c10::str(Dropout2d(0.42)), "smith::nn::Dropout2d(p=0.42, inplace=false)");
  ASSERT_EQ(
      c10::str(Dropout2d(Dropout2dOptions().p(0.42).inplace(true))),
      "smith::nn::Dropout2d(p=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintDropout3d) {
  ASSERT_EQ(
      c10::str(Dropout3d()), "smith::nn::Dropout3d(p=0.5, inplace=false)");
  ASSERT_EQ(
      c10::str(Dropout3d(0.42)), "smith::nn::Dropout3d(p=0.42, inplace=false)");
  ASSERT_EQ(
      c10::str(Dropout3d(Dropout3dOptions().p(0.42).inplace(true))),
      "smith::nn::Dropout3d(p=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintFunctional) {
  ASSERT_EQ(c10::str(Functional(smith::relu)), "smith::nn::Functional()");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm1d) {
  ASSERT_EQ(
      c10::str(BatchNorm1d(BatchNorm1dOptions(4)
                               .eps(0.5)
                               .momentum(0.1)
                               .affine(false)
                               .track_running_stats(true))),
      "smith::nn::BatchNorm1d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm2d) {
  ASSERT_EQ(
      c10::str(BatchNorm2d(BatchNorm2dOptions(4)
                               .eps(0.5)
                               .momentum(0.1)
                               .affine(false)
                               .track_running_stats(true))),
      "smith::nn::BatchNorm2d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintBatchNorm3d) {
  ASSERT_EQ(
      c10::str(BatchNorm3d(BatchNorm3dOptions(4)
                               .eps(0.5)
                               .momentum(0.1)
                               .affine(false)
                               .track_running_stats(true))),
      "smith::nn::BatchNorm3d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintInstanceNorm1d) {
  ASSERT_EQ(
      c10::str(InstanceNorm1d(InstanceNorm1dOptions(4)
                                  .eps(0.5)
                                  .momentum(0.1)
                                  .affine(false)
                                  .track_running_stats(true))),
      "smith::nn::InstanceNorm1d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintInstanceNorm2d) {
  ASSERT_EQ(
      c10::str(InstanceNorm2d(InstanceNorm2dOptions(4)
                                  .eps(0.5)
                                  .momentum(0.1)
                                  .affine(false)
                                  .track_running_stats(true))),
      "smith::nn::InstanceNorm2d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintInstanceNorm3d) {
  ASSERT_EQ(
      c10::str(InstanceNorm3d(InstanceNorm3dOptions(4)
                                  .eps(0.5)
                                  .momentum(0.1)
                                  .affine(false)
                                  .track_running_stats(true))),
      "smith::nn::InstanceNorm3d(4, eps=0.5, momentum=0.1, affine=false, track_running_stats=true)");
}

TEST_F(ModulesTest, PrettyPrintLayerNorm) {
  ASSERT_EQ(
      c10::str(LayerNorm(LayerNormOptions({2, 2}))),
      "smith::nn::LayerNorm([2, 2], eps=1e-05, elementwise_affine=true)");
  ASSERT_EQ(
      c10::str(LayerNorm(
          LayerNormOptions({2, 2}).elementwise_affine(false).eps(2e-5))),
      "smith::nn::LayerNorm([2, 2], eps=2e-05, elementwise_affine=false)");
}

TEST_F(ModulesTest, PrettyPrintGroupNorm) {
  ASSERT_EQ(
      c10::str(GroupNorm(GroupNormOptions(2, 2))),
      "smith::nn::GroupNorm(2, 2, eps=1e-05, affine=true)");
  ASSERT_EQ(
      c10::str(GroupNorm(GroupNormOptions(2, 2).eps(2e-5).affine(false))),
      "smith::nn::GroupNorm(2, 2, eps=2e-05, affine=false)");
}

TEST_F(ModulesTest, PrettyPrintLocalResponseNorm) {
  ASSERT_EQ(
      c10::str(LocalResponseNorm(LocalResponseNormOptions(2))),
      "smith::nn::LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=1)");
  ASSERT_EQ(
      c10::str(LocalResponseNorm(
          LocalResponseNormOptions(2).alpha(0.0002).beta(0.85).k(2.))),
      "smith::nn::LocalResponseNorm(2, alpha=0.0002, beta=0.85, k=2)");
}

TEST_F(ModulesTest, PrettyPrintEmbedding) {
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2))),
      "smith::nn::Embedding(num_embeddings=10, embedding_dim=2)");
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2).padding_idx(3).max_norm(2))),
      "smith::nn::Embedding(num_embeddings=10, embedding_dim=2, padding_idx=3, max_norm=2)");
  ASSERT_EQ(
      c10::str(Embedding(EmbeddingOptions(10, 2)
                             .padding_idx(3)
                             .max_norm(2)
                             .norm_type(2.5)
                             .scale_grad_by_freq(true)
                             .sparse(true))),
      "smith::nn::Embedding(num_embeddings=10, embedding_dim=2, padding_idx=3, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true)");
}

TEST_F(ModulesTest, PrettyPrintEmbeddingBag) {
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2))),
      "smith::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2).max_norm(2))),
      "smith::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2)
                                .max_norm(2)
                                .norm_type(2.5)
                                .scale_grad_by_freq(true)
                                .sparse(true))),
      "smith::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2)
                                .max_norm(2)
                                .norm_type(2.5)
                                .scale_grad_by_freq(true)
                                .sparse(true)
                                .mode(smith::kSum))),
      "smith::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true, mode=kSum)");
  ASSERT_EQ(
      c10::str(EmbeddingBag(EmbeddingBagOptions(10, 2)
                                .max_norm(2)
                                .norm_type(2.5)
                                .scale_grad_by_freq(true)
                                .sparse(true)
                                .mode(smith::kSum)
                                .padding_idx(5))),
      "smith::nn::EmbeddingBag(num_embeddings=10, embedding_dim=2, max_norm=2, norm_type=2.5, scale_grad_by_freq=true, sparse=true, mode=kSum, padding_idx=5)");
}

TEST_F(ModulesTest, PrettyPrintL1Loss) {
  ASSERT_EQ(c10::str(L1Loss()), "smith::nn::L1Loss()");
}
TEST_F(ModulesTest, PrettyPrintKLDivLoss) {
  ASSERT_EQ(c10::str(KLDivLoss()), "smith::nn::KLDivLoss()");
}
TEST_F(ModulesTest, PrettyPrintMSELoss) {
  ASSERT_EQ(c10::str(MSELoss()), "smith::nn::MSELoss()");
}
TEST_F(ModulesTest, PrettyPrintBCELoss) {
  ASSERT_EQ(c10::str(BCELoss()), "smith::nn::BCELoss()");
}
TEST_F(ModulesTest, PrettyPrintHingeEmbeddingLoss) {
  ASSERT_EQ(
      c10::str(HingeEmbeddingLoss(HingeEmbeddingLossOptions().margin(4))),
      "smith::nn::HingeEmbeddingLoss(margin=4)");
}

TEST_F(ModulesTest, PrettyPrintCosineEmbeddingLoss) {
  ASSERT_EQ(
      c10::str(CosineEmbeddingLoss(CosineEmbeddingLossOptions().margin(0.25))),
      "smith::nn::CosineEmbeddingLoss(margin=0.25)");
}

TEST_F(ModulesTest, PrettyPrintTripletMarginLoss) {
  ASSERT_EQ(
      c10::str(TripletMarginLoss(
          TripletMarginLossOptions().margin(3).p(2).eps(1e-06).swap(false))),
      "smith::nn::TripletMarginLoss(margin=3, p=2, eps=1e-06, swap=false)");
}

TEST_F(ModulesTest, PrettyPrintTripletMarginWithDistanceLoss) {
  auto distanceOptions = TripletMarginWithDistanceLossOptions()
                             .distance_function([&](const smith::Tensor& x,
                                                    const smith::Tensor& y) {
                               return smith::pairwise_distance(x, y, 2.0, 1e-6);
                             })
                             .margin(1.5)
                             .swap(true)
                             .reduction(smith::kMean);
  ASSERT_EQ(
      c10::str(TripletMarginWithDistanceLoss(distanceOptions)),
      "smith::nn::TripletMarginWithDistanceLoss(margin=1.5, swap=true)");
}

TEST_F(ModulesTest, PrettyPrintNLLLoss) {
  ASSERT_EQ(c10::str(NLLLoss()), "smith::nn::NLLLoss()");
}

TEST_F(ModulesTest, PrettyPrinCrossEntropyLoss) {
  ASSERT_EQ(c10::str(CrossEntropyLoss()), "smith::nn::CrossEntropyLoss()");
}

TEST_F(ModulesTest, PrettyPrintMultiLabelMarginLoss) {
  ASSERT_EQ(
      c10::str(MultiLabelMarginLoss()), "smith::nn::MultiLabelMarginLoss()");
}

TEST_F(ModulesTest, PrettyPrintMultiLabelSoftMarginLoss) {
  ASSERT_EQ(
      c10::str(MultiLabelSoftMarginLoss()),
      "smith::nn::MultiLabelSoftMarginLoss()");
}

TEST_F(ModulesTest, PrettyPrintSoftMarginLoss) {
  ASSERT_EQ(c10::str(SoftMarginLoss()), "smith::nn::SoftMarginLoss()");
}

TEST_F(ModulesTest, PrettyPrintCosineSimilarity) {
  ASSERT_EQ(
      c10::str(CosineSimilarity()),
      "smith::nn::CosineSimilarity(dim=1, eps=1e-08)");
  ASSERT_EQ(
      c10::str(CosineSimilarity(CosineSimilarityOptions().dim(0).eps(0.5))),
      "smith::nn::CosineSimilarity(dim=0, eps=0.5)");
}

TEST_F(ModulesTest, PrettyPrintPairwiseDistance) {
  ASSERT_EQ(
      c10::str(PairwiseDistance()),
      "smith::nn::PairwiseDistance(p=2, eps=1e-06, keepdim=false)");
  ASSERT_EQ(
      c10::str(PairwiseDistance(
          PairwiseDistanceOptions().p(3).eps(0.5).keepdim(true))),
      "smith::nn::PairwiseDistance(p=3, eps=0.5, keepdim=true)");
}

TEST_F(ModulesTest, PrettyPrintReflectionPad) {
  ASSERT_EQ(
      c10::str(ReflectionPad1d(ReflectionPad1dOptions(2))),
      "smith::nn::ReflectionPad1d(padding=[2, 2])");
  ASSERT_EQ(
      c10::str(ReflectionPad1d(ReflectionPad1dOptions({3, 1}))),
      "smith::nn::ReflectionPad1d(padding=[3, 1])");
  ASSERT_EQ(
      c10::str(ReflectionPad2d(ReflectionPad2dOptions(2))),
      "smith::nn::ReflectionPad2d(padding=[2, 2, 2, 2])");
  ASSERT_EQ(
      c10::str(ReflectionPad2d(ReflectionPad2dOptions({1, 1, 2, 0}))),
      "smith::nn::ReflectionPad2d(padding=[1, 1, 2, 0])");
}

TEST_F(ModulesTest, PrettyPrintReplicationPad) {
  ASSERT_EQ(
      c10::str(ReplicationPad1d(ReplicationPad1dOptions(2))),
      "smith::nn::ReplicationPad1d(padding=[2, 2])");
  ASSERT_EQ(
      c10::str(ReplicationPad1d(ReplicationPad1dOptions({3, 1}))),
      "smith::nn::ReplicationPad1d(padding=[3, 1])");
  ASSERT_EQ(
      c10::str(ReplicationPad2d(ReplicationPad2dOptions(2))),
      "smith::nn::ReplicationPad2d(padding=[2, 2, 2, 2])");
  ASSERT_EQ(
      c10::str(ReplicationPad2d(ReplicationPad2dOptions({1, 1, 2, 0}))),
      "smith::nn::ReplicationPad2d(padding=[1, 1, 2, 0])");
  ASSERT_EQ(
      c10::str(ReplicationPad3d(ReplicationPad3dOptions(1))),
      "smith::nn::ReplicationPad3d(padding=[1, 1, 1, 1, 1, 1])");
  ASSERT_EQ(
      c10::str(ReplicationPad3d(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}))),
      "smith::nn::ReplicationPad3d(padding=[1, 2, 1, 2, 1, 2])");
}

TEST_F(ModulesTest, PrettyPrintZeroPad) {
  ASSERT_EQ(
      c10::str(ZeroPad1d(ZeroPad1dOptions(2))),
      "smith::nn::ZeroPad1d(padding=[2, 2])");
  ASSERT_EQ(
      c10::str(ZeroPad1d(ZeroPad1dOptions({3, 1}))),
      "smith::nn::ZeroPad1d(padding=[3, 1])");
  ASSERT_EQ(
      c10::str(ZeroPad2d(ZeroPad2dOptions(2))),
      "smith::nn::ZeroPad2d(padding=[2, 2, 2, 2])");
  ASSERT_EQ(
      c10::str(ZeroPad2d(ZeroPad2dOptions({1, 1, 2, 0}))),
      "smith::nn::ZeroPad2d(padding=[1, 1, 2, 0])");
  ASSERT_EQ(
      c10::str(ZeroPad3d(ZeroPad3dOptions(1))),
      "smith::nn::ZeroPad3d(padding=[1, 1, 1, 1, 1, 1])");
  ASSERT_EQ(
      c10::str(ZeroPad3d(ZeroPad3dOptions({1, 2, 1, 2, 1, 2}))),
      "smith::nn::ZeroPad3d(padding=[1, 2, 1, 2, 1, 2])");
}

TEST_F(ModulesTest, PrettyPrintConstantPad) {
  ASSERT_EQ(
      c10::str(ConstantPad1d(ConstantPad1dOptions(2, 3.5))),
      "smith::nn::ConstantPad1d(padding=[2, 2], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad1d(ConstantPad1dOptions({3, 1}, 3.5))),
      "smith::nn::ConstantPad1d(padding=[3, 1], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad2d(ConstantPad2dOptions(2, 3.5))),
      "smith::nn::ConstantPad2d(padding=[2, 2, 2, 2], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad2d(ConstantPad2dOptions({3, 0, 2, 1}, 3.5))),
      "smith::nn::ConstantPad2d(padding=[3, 0, 2, 1], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad3d(ConstantPad3dOptions(1, 3.5))),
      "smith::nn::ConstantPad3d(padding=[1, 1, 1, 1, 1, 1], value=3.5)");
  ASSERT_EQ(
      c10::str(ConstantPad3d(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5))),
      "smith::nn::ConstantPad3d(padding=[1, 2, 1, 2, 1, 2], value=3.5)");
}

TEST_F(ModulesTest, PrettyPrintNestedModel) {
  struct InnerTestModule : smith::nn::Module {
    InnerTestModule()
        : smith::nn::Module("InnerTestModule"),
          fc(register_module("fc", smith::nn::Linear(3, 4))),
          table(register_module("table", smith::nn::Embedding(10, 2))) {}

    smith::nn::Linear fc;
    smith::nn::Embedding table;
  };

  struct TestModule : smith::nn::Module {
    TestModule()
        : smith::nn::Module("TestModule"),
          fc(register_module("fc", smith::nn::Linear(4, 5))),
          table(register_module(
              "table",
              smith::nn::Embedding(EmbeddingOptions(10, 2)))),
          inner(register_module("inner", std::make_shared<InnerTestModule>())) {
    }

    smith::nn::Linear fc;
    smith::nn::Embedding table;
    std::shared_ptr<InnerTestModule> inner;
  };

  ASSERT_EQ(
      c10::str(TestModule{}),
      "TestModule(\n"
      "  (fc): smith::nn::Linear(in_features=4, out_features=5, bias=true)\n"
      "  (table): smith::nn::Embedding(num_embeddings=10, embedding_dim=2)\n"
      "  (inner): InnerTestModule(\n"
      "    (fc): smith::nn::Linear(in_features=3, out_features=4, bias=true)\n"
      "    (table): smith::nn::Embedding(num_embeddings=10, embedding_dim=2)\n"
      "  )\n"
      ")");
}

TEST_F(ModulesTest, PrettyPrintELU) {
  ASSERT_EQ(c10::str(ELU()), "smith::nn::ELU(alpha=1)");
  ASSERT_EQ(
      c10::str(ELU(ELUOptions().alpha(42.42).inplace(true))),
      "smith::nn::ELU(alpha=42.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintSELU) {
  ASSERT_EQ(c10::str(SELU()), "smith::nn::SELU()");
  ASSERT_EQ(
      c10::str(SELU(SELUOptions().inplace(true))),
      "smith::nn::SELU(inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintGLU) {
  ASSERT_EQ(c10::str(GLU()), "smith::nn::GLU(dim=-1)");
  ASSERT_EQ(c10::str(GLU(1)), "smith::nn::GLU(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintHardshrink) {
  ASSERT_EQ(c10::str(Hardshrink()), "smith::nn::Hardshrink(0.5)");
  ASSERT_EQ(
      c10::str(Hardshrink(HardshrinkOptions().lambda(42.42))),
      "smith::nn::Hardshrink(42.42)");
}

TEST_F(ModulesTest, PrettyPrintHardtanh) {
  ASSERT_EQ(c10::str(Hardtanh()), "smith::nn::Hardtanh(min_val=-1, max_val=1)");
  ASSERT_EQ(
      c10::str(Hardtanh(
          HardtanhOptions().min_val(-42.42).max_val(0.42).inplace(true))),
      "smith::nn::Hardtanh(min_val=-42.42, max_val=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintLeakyReLU) {
  ASSERT_EQ(c10::str(LeakyReLU()), "smith::nn::LeakyReLU(negative_slope=0.01)");
  ASSERT_EQ(
      c10::str(
          LeakyReLU(LeakyReLUOptions().negative_slope(0.42).inplace(true))),
      "smith::nn::LeakyReLU(negative_slope=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintLogSigmoid) {
  ASSERT_EQ(c10::str(LogSigmoid()), "smith::nn::LogSigmoid()");
}

TEST_F(ModulesTest, PrettyPrintSoftmax) {
  ASSERT_EQ(c10::str(Softmax(SoftmaxOptions(1))), "smith::nn::Softmax(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintSoftmin) {
  ASSERT_EQ(c10::str(Softmin(SoftminOptions(1))), "smith::nn::Softmin(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintLogSoftmax) {
  ASSERT_EQ(
      c10::str(LogSoftmax(LogSoftmaxOptions(1))),
      "smith::nn::LogSoftmax(dim=1)");
}

TEST_F(ModulesTest, PrettyPrintSoftmax2d) {
  ASSERT_EQ(c10::str(Softmax2d()), "smith::nn::Softmax2d()");
}

TEST_F(ModulesTest, PrettyPrintPReLU) {
  ASSERT_EQ(c10::str(PReLU()), "smith::nn::PReLU(num_parameters=1)");
  ASSERT_EQ(
      c10::str(PReLU(PReLUOptions().num_parameters(42))),
      "smith::nn::PReLU(num_parameters=42)");
}

TEST_F(ModulesTest, PrettyPrintReLU) {
  ASSERT_EQ(c10::str(ReLU()), "smith::nn::ReLU()");
  ASSERT_EQ(
      c10::str(ReLU(ReLUOptions().inplace(true))),
      "smith::nn::ReLU(inplace=true)");
  ASSERT_EQ(c10::str(ReLU(/*inplace=*/true)), "smith::nn::ReLU(inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintReLU6) {
  ASSERT_EQ(c10::str(ReLU6()), "smith::nn::ReLU6()");
  ASSERT_EQ(
      c10::str(ReLU6(ReLU6Options().inplace(true))),
      "smith::nn::ReLU6(inplace=true)");
  ASSERT_EQ(
      c10::str(ReLU6(/*inplace=*/true)), "smith::nn::ReLU6(inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintRReLU) {
  ASSERT_EQ(c10::str(RReLU()), "smith::nn::RReLU(lower=0.125, upper=0.333333)");
  ASSERT_EQ(
      c10::str(RReLU(RReLUOptions().lower(0.24).upper(0.42).inplace(true))),
      "smith::nn::RReLU(lower=0.24, upper=0.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintCELU) {
  ASSERT_EQ(c10::str(CELU()), "smith::nn::CELU(alpha=1)");
  ASSERT_EQ(
      c10::str(CELU(CELUOptions().alpha(42.42).inplace(true))),
      "smith::nn::CELU(alpha=42.42, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintSigmoid) {
  ASSERT_EQ(c10::str(Sigmoid()), "smith::nn::Sigmoid()");
}

TEST_F(ModulesTest, PrettyPrintPixelShuffle) {
  ASSERT_EQ(
      c10::str(PixelShuffle(PixelShuffleOptions(5))),
      "smith::nn::PixelShuffle(upscale_factor=5)");
}

TEST_F(ModulesTest, PrettyPrintPixelUnshuffle) {
  ASSERT_EQ(
      c10::str(PixelUnshuffle(PixelUnshuffleOptions(5))),
      "smith::nn::PixelUnshuffle(downscale_factor=5)");
}

TEST_F(ModulesTest, PrettyPrintSoftplus) {
  ASSERT_EQ(c10::str(Softplus()), "smith::nn::Softplus(beta=1, threshold=20)");
  ASSERT_EQ(
      c10::str(Softplus(SoftplusOptions().beta(0.24).threshold(42.42))),
      "smith::nn::Softplus(beta=0.24, threshold=42.42)");
}

TEST_F(ModulesTest, PrettyPrintSoftshrink) {
  ASSERT_EQ(c10::str(Softshrink()), "smith::nn::Softshrink(0.5)");
  ASSERT_EQ(
      c10::str(Softshrink(SoftshrinkOptions(42.42))),
      "smith::nn::Softshrink(42.42)");
}

TEST_F(ModulesTest, PrettyPrintSoftsign) {
  ASSERT_EQ(c10::str(Softsign()), "smith::nn::Softsign()");
}

TEST_F(ModulesTest, PrettyPrintTanh) {
  ASSERT_EQ(c10::str(Tanh()), "smith::nn::Tanh()");
}

TEST_F(ModulesTest, PrettyPrintTanhshrink) {
  ASSERT_EQ(c10::str(Tanhshrink()), "smith::nn::Tanhshrink()");
}

TEST_F(ModulesTest, PrettyPrintThreshold) {
  ASSERT_EQ(
      c10::str(Threshold(24.24, 42.42)),
      "smith::nn::Threshold(threshold=24.24, value=42.42)");
  ASSERT_EQ(
      c10::str(Threshold(ThresholdOptions(42.42, 24.24).inplace(true))),
      "smith::nn::Threshold(threshold=42.42, value=24.24, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintCTCLoss) {
  ASSERT_EQ(c10::str(CTCLoss()), "smith::nn::CTCLoss()");
  ASSERT_EQ(
      c10::str(
          CTCLoss(CTCLossOptions().blank(42).zero_infinity(false).reduction(
              smith::kSum))),
      "smith::nn::CTCLoss()");
}

TEST_F(ModulesTest, PrettyPrintPoissonNLLLoss) {
  ASSERT_EQ(c10::str(PoissonNLLLoss()), "smith::nn::PoissonNLLLoss()");
  ASSERT_EQ(
      c10::str(PoissonNLLLoss(PoissonNLLLossOptions()
                                  .log_input(false)
                                  .full(true)
                                  .eps(0.42)
                                  .reduction(smith::kSum))),
      "smith::nn::PoissonNLLLoss()");
}

TEST_F(ModulesTest, PrettyPrintMarginRankingLoss) {
  ASSERT_EQ(c10::str(MarginRankingLoss()), "smith::nn::MarginRankingLoss()");
  ASSERT_EQ(
      c10::str(MarginRankingLoss(
          MarginRankingLossOptions().margin(0.5).reduction(smith::kSum))),
      "smith::nn::MarginRankingLoss()");
}

TEST_F(ModulesTest, PrettyPrintCrossMapLRN2d) {
  ASSERT_EQ(
      c10::str(CrossMapLRN2d(4)),
      "smith::nn::CrossMapLRN2d(4, alpha=0.0001, beta=0.75, k=1)");
  ASSERT_EQ(
      c10::str(
          CrossMapLRN2d(CrossMapLRN2dOptions(3).alpha(1e-5).beta(0.1).k(10))),
      "smith::nn::CrossMapLRN2d(3, alpha=1e-05, beta=0.1, k=10)");
}

TEST_F(ModulesTest, PrettyPrintAlphaDropout) {
  ASSERT_EQ(
      c10::str(AlphaDropout()),
      "smith::nn::AlphaDropout(p=0.5, inplace=false)");
  ASSERT_EQ(
      c10::str(AlphaDropout(AlphaDropoutOptions(0.2))),
      "smith::nn::AlphaDropout(p=0.2, inplace=false)");
  ASSERT_EQ(
      c10::str(AlphaDropout(AlphaDropoutOptions(0.2).inplace(true))),
      "smith::nn::AlphaDropout(p=0.2, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintFeatureAlphaDropout) {
  ASSERT_EQ(
      c10::str(FeatureAlphaDropout()),
      "smith::nn::FeatureAlphaDropout(p=0.5, inplace=false)");
  ASSERT_EQ(
      c10::str(FeatureAlphaDropout(FeatureAlphaDropoutOptions(0.2))),
      "smith::nn::FeatureAlphaDropout(p=0.2, inplace=false)");
  ASSERT_EQ(
      c10::str(
          FeatureAlphaDropout(FeatureAlphaDropoutOptions(0.2).inplace(true))),
      "smith::nn::FeatureAlphaDropout(p=0.2, inplace=true)");
}

TEST_F(ModulesTest, PrettyPrintBCEWithLogitsLoss) {
  ASSERT_EQ(c10::str(BCEWithLogitsLoss()), "smith::nn::BCEWithLogitsLoss()");
  ASSERT_EQ(
      c10::str(BCEWithLogitsLoss(BCEWithLogitsLossOptions()
                                     .weight(smith::ones({3, 3}))
                                     .pos_weight(smith::ones({3, 3}))
                                     .reduction(smith::kSum))),
      "smith::nn::BCEWithLogitsLoss()");
}

TEST_F(ModulesTest, PrettyPrintMultiheadAttention) {
  ASSERT_EQ(
      c10::str(MultiheadAttention(20, 10)),
      "smith::nn::MultiheadAttention(\n  (out_proj): smith::nn::Linear(in_features=20, out_features=20, bias=true)\n)");
  ASSERT_EQ(
      c10::str(
          MultiheadAttention(MultiheadAttentionOptions(20, 10).bias(false))),
      "smith::nn::MultiheadAttention(\n  (out_proj): smith::nn::Linear(in_features=20, out_features=20, bias=false)\n)");
}

TEST_F(ModulesTest, PrettyPrintRNNCell) {
  ASSERT_EQ(c10::str(RNNCell(20, 10)), "smith::nn::RNNCell(20, 10)");
  ASSERT_EQ(
      c10::str(RNNCell(
          RNNCellOptions(20, 10).bias(false).nonlinearity(smith::kTanh))),
      "smith::nn::RNNCell(20, 10, bias=false)");
  ASSERT_EQ(
      c10::str(RNNCell(
          RNNCellOptions(20, 10).bias(false).nonlinearity(smith::kReLU))),
      "smith::nn::RNNCell(20, 10, bias=false, nonlinearity=kReLU)");
}

TEST_F(ModulesTest, PrettyPrintLSTMCell) {
  ASSERT_EQ(c10::str(LSTMCell(20, 10)), "smith::nn::LSTMCell(20, 10)");
  ASSERT_EQ(
      c10::str(LSTMCell(LSTMCellOptions(20, 10).bias(false))),
      "smith::nn::LSTMCell(20, 10, bias=false)");
}

TEST_F(ModulesTest, PrettyPrintGRUCell) {
  ASSERT_EQ(c10::str(GRUCell(20, 10)), "smith::nn::GRUCell(20, 10)");
  ASSERT_EQ(
      c10::str(GRUCell(GRUCellOptions(20, 10).bias(false))),
      "smith::nn::GRUCell(20, 10, bias=false)");
}

TEST_F(ModulesTest, PrettyPrintAdaptiveLogSoftmaxWithLoss) {
  {
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 4, {2}).div_value(2.));
    ASSERT_EQ(
        c10::str(asfm),
        "smith::nn::AdaptiveLogSoftmaxWithLoss(\n"
        "  (head): smith::nn::Linear(in_features=8, out_features=3, bias=false)\n"
        "  (tail): smith::nn::ModuleList(\n"
        "    (0): smith::nn::Sequential(\n"
        "      (0): smith::nn::Linear(in_features=8, out_features=4, bias=false)\n"
        "      (1): smith::nn::Linear(in_features=4, out_features=2, bias=false)\n"
        "    )\n"
        "  )\n"
        ")");
  }
  {
    AdaptiveLogSoftmaxWithLoss asfm(
        AdaptiveLogSoftmaxWithLossOptions(8, 10, {4, 8})
            .div_value(2.)
            .head_bias(true));
    ASSERT_EQ(
        c10::str(asfm),
        "smith::nn::AdaptiveLogSoftmaxWithLoss(\n"
        "  (head): smith::nn::Linear(in_features=8, out_features=6, bias=true)\n"
        "  (tail): smith::nn::ModuleList(\n"
        "    (0): smith::nn::Sequential(\n"
        "      (0): smith::nn::Linear(in_features=8, out_features=4, bias=false)\n"
        "      (1): smith::nn::Linear(in_features=4, out_features=4, bias=false)\n"
        "    )\n"
        "    (1): smith::nn::Sequential(\n"
        "      (0): smith::nn::Linear(in_features=8, out_features=2, bias=false)\n"
        "      (1): smith::nn::Linear(in_features=2, out_features=2, bias=false)\n"
        "    )\n"
        "  )\n"
        ")");
  }
}
