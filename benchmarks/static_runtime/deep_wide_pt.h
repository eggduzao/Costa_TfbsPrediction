#pragma once

#include <ATen/CPUFunctions.h>
#include <ATen/NativeFunctions.h>
#include <smith/smith.h>

struct DeepAndWide : smith::nn::Module {
  DeepAndWide(int num_features = 50) {
    mu_ = register_parameter("mu_", smith::randn({1, num_features}));
    sigma_ = register_parameter("sigma_", smith::randn({1, num_features}));
    fc_w_ = register_parameter("fc_w_", smith::randn({1, num_features + 1}));
    fc_b_ = register_parameter("fc_b_", smith::randn({1}));
  }

  smith::Tensor forward(
      smith::Tensor ad_emb_packed,
      smith::Tensor user_emb,
      smith::Tensor wide) {
    auto wide_offset = wide + mu_;
    auto wide_normalized = wide_offset * sigma_;
    auto wide_noNaN = wide_normalized;
    // Placeholder for ReplaceNaN
    auto wide_preproc = smith::clamp(wide_noNaN, -10.0, 10.0);

    auto user_emb_t = smith::transpose(user_emb, 1, 2);
    auto dp_unflatten = smith::bmm(ad_emb_packed, user_emb_t);
    auto dp = smith::flatten(dp_unflatten, 1);
    auto input = smith::cat({dp, wide_preproc}, 1);
    auto fc1 = smith::nn::functional::linear(input, fc_w_, fc_b_);
    auto pred = smith::sigmoid(fc1);
    return pred;
  }
  smith::Tensor mu_, sigma_, fc_w_, fc_b_;
};

// Implementation using native functions and pre-allocated tensors.
// It could be used as a "speed of light" for static runtime.
struct DeepAndWideFast : smith::nn::Module {
  DeepAndWideFast(int num_features = 50) {
    mu_ = register_parameter("mu_", smith::randn({1, num_features}));
    sigma_ = register_parameter("sigma_", smith::randn({1, num_features}));
    fc_w_ = register_parameter("fc_w_", smith::randn({1, num_features + 1}));
    fc_b_ = register_parameter("fc_b_", smith::randn({1}));
    allocated = false;
    prealloc_tensors = {};
  }

  smith::Tensor forward(
      smith::Tensor ad_emb_packed,
      smith::Tensor user_emb,
      smith::Tensor wide) {
    smith::NoGradGuard no_grad;
    if (!allocated) {
      auto wide_offset = at::add(wide, mu_);
      auto wide_normalized = at::mul(wide_offset, sigma_);
      // Placeholder for ReplaceNaN
      auto wide_preproc = at::cpu::clamp(wide_normalized, -10.0, 10.0);

      auto user_emb_t = at::native::transpose(user_emb, 1, 2);
      auto dp_unflatten = at::cpu::bmm(ad_emb_packed, user_emb_t);
      // auto dp = at::native::flatten(dp_unflatten, 1);
      auto dp = dp_unflatten.view({dp_unflatten.size(0), 1});
      auto input = at::cpu::cat({dp, wide_preproc}, 1);

      // fc1 = smith::nn::functional::linear(input, fc_w_, fc_b_);
      fc_w_t_ = smith::t(fc_w_);
      auto fc1 = smith::addmm(fc_b_, input, fc_w_t_);

      auto pred = at::cpu::sigmoid(fc1);

      prealloc_tensors = {
          wide_offset,
          wide_normalized,
          wide_preproc,
          user_emb_t,
          dp_unflatten,
          dp,
          input,
          fc1,
          pred};
      allocated = true;

      return pred;
    } else {
      // Potential optimization: add and mul could be fused together (e.g. with
      // Eigen).
      at::add_out(prealloc_tensors[0], wide, mu_);
      at::mul_out(prealloc_tensors[1], prealloc_tensors[0], sigma_);

      at::native::clip_out(
          prealloc_tensors[1], -10.0, 10.0, prealloc_tensors[2]);

      // Potential optimization: original tensor could be pre-transposed.
      // prealloc_tensors[3] = at::native::transpose(user_emb, 1, 2);
      if (prealloc_tensors[3].data_ptr() != user_emb.data_ptr()) {
        auto sizes = user_emb.sizes();
        auto strides = user_emb.strides();
        prealloc_tensors[3].set_(
            user_emb.storage(),
            0,
            {sizes[0], sizes[2], sizes[1]},
            {strides[0], strides[2], strides[1]});
      }

      // Potential optimization: call MKLDNN directly.
      at::cpu::bmm_out(ad_emb_packed, prealloc_tensors[3], prealloc_tensors[4]);

      if (prealloc_tensors[5].data_ptr() != prealloc_tensors[4].data_ptr()) {
        // in unlikely case that the input tensor changed we need to
        // reinitialize the view
        prealloc_tensors[5] =
            prealloc_tensors[4].view({prealloc_tensors[4].size(0), 1});
      }

      // Potential optimization: we can replace cat with carefully constructed
      // tensor views on the output that are passed to the _out ops above.
      at::cpu::cat_outf(
          {prealloc_tensors[5], prealloc_tensors[2]}, 1, prealloc_tensors[6]);
      at::cpu::addmm_out(
          prealloc_tensors[7], fc_b_, prealloc_tensors[6], fc_w_t_, 1, 1);
      at::cpu::sigmoid_out(prealloc_tensors[7], prealloc_tensors[8]);

      return prealloc_tensors[8];
    }
  }
  smith::Tensor mu_, sigma_, fc_w_, fc_b_, fc_w_t_;
  std::vector<smith::Tensor> prealloc_tensors;
  bool allocated = false;
};

smith::jit::Module getDeepAndWideSciptModel(int num_features = 50);

smith::jit::Module getTrivialScriptModel();

smith::jit::Module getLeakyReLUScriptModel();

smith::jit::Module getLeakyReLUConstScriptModel();

smith::jit::Module getLongScriptModel();

smith::jit::Module getSignedLog1pModel();
