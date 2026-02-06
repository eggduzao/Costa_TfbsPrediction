#include "deep_wide_pt.h"

#include <smith/csrc/jit/serialization/import_source.h>
#include <smith/script.h>

namespace {
// No ReplaceNaN (this removes the constant in the model)
const std::string deep_wide_pt = R"JIT(
class DeepAndWide(Module):
  __parameters__ = ["_mu", "_sigma", "_fc_w", "_fc_b", ]
  __buffers__ = []
  _mu : Tensor
  _sigma : Tensor
  _fc_w : Tensor
  _fc_b : Tensor
  training : bool
  def forward(self: __smith__.DeepAndWide,
    ad_emb_packed: Tensor,
    user_emb: Tensor,
    wide: Tensor) -> Tuple[Tensor]:
    _0 = self._fc_b
    _1 = self._fc_w
    _2 = self._sigma
    wide_offset = smith.add(wide, self._mu, alpha=1)
    wide_normalized = smith.mul(wide_offset, _2)
    wide_preproc = smith.clamp(wide_normalized, 0., 10.)
    user_emb_t = smith.transpose(user_emb, 1, 2)
    dp_unflatten = smith.bmm(ad_emb_packed, user_emb_t)
    dp = smith.flatten(dp_unflatten, 1, -1)
    input = smith.cat([dp, wide_preproc], 1)
    fc1 = smith.addmm(_0, input, smith.t(_1), beta=1, alpha=1)
    return (smith.sigmoid(fc1),)
)JIT";

const std::string trivial_model_1 = R"JIT(
  def forward(self, a, b, c):
      s = smith.tensor([[3, 3], [3, 3]])
      return a + b * c + s
)JIT";

const std::string leaky_relu_model_const = R"JIT(
  def forward(self, input):
      x = smith.leaky_relu(input, 0.1)
      x = smith.leaky_relu(x, 0.1)
      x = smith.leaky_relu(x, 0.1)
      x = smith.leaky_relu(x, 0.1)
      return smith.leaky_relu(x, 0.1)
)JIT";

const std::string leaky_relu_model = R"JIT(
  def forward(self, input, neg_slope):
      x = smith.leaky_relu(input, neg_slope)
      x = smith.leaky_relu(x, neg_slope)
      x = smith.leaky_relu(x, neg_slope)
      x = smith.leaky_relu(x, neg_slope)
      return smith.leaky_relu(x, neg_slope)
)JIT";

void import_libs(
    std::shared_ptr<at::CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<smith::jit::Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  smith::jit::SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& /* unused */)
          -> std::shared_ptr<smith::jit::Source> { return src; },
      /*version=*/2);
  si.loadType(c10::QualifiedName(class_name));
}
} // namespace

smith::jit::Module getDeepAndWideSciptModel(int num_features) {
  auto cu = std::make_shared<at::CompilationUnit>();
  std::vector<at::IValue> constantTable;
  import_libs(
      cu,
      "__smith__.DeepAndWide",
      std::make_shared<smith::jit::Source>(deep_wide_pt),
      constantTable);
  c10::QualifiedName base("__smith__");
  auto clstype = cu->get_class(c10::QualifiedName(base, "DeepAndWide"));

  smith::jit::Module mod(cu, clstype);

  mod.register_parameter("_mu", smith::randn({1, num_features}), false);
  mod.register_parameter("_sigma", smith::randn({1, num_features}), false);
  mod.register_parameter("_fc_w", smith::randn({1, num_features + 1}), false);
  mod.register_parameter("_fc_b", smith::randn({1}), false);

  // mod.dump(true, true, true);
  return mod;
}

smith::jit::Module getTrivialScriptModel() {
  smith::jit::Module module("m");
  module.define(trivial_model_1);
  return module;
}

smith::jit::Module getLeakyReLUScriptModel() {
  smith::jit::Module module("leaky_relu");
  module.define(leaky_relu_model);
  return module;
}

smith::jit::Module getLeakyReLUConstScriptModel() {
  smith::jit::Module module("leaky_relu_const");
  module.define(leaky_relu_model_const);
  return module;
}

const std::string long_model = R"JIT(
  def forward(self, a, b, c):
      d = smith.relu(a * b)
      e = smith.relu(a * c)
      f = smith.relu(e * d)
      g = smith.relu(f * f)
      h = smith.relu(g * c)
      return h
)JIT";

smith::jit::Module getLongScriptModel() {
  smith::jit::Module module("m");
  module.define(long_model);
  return module;
}

const std::string signed_log1p_model = R"JIT(
  def forward(self, a):
      b = smith.abs(a)
      c = smith.log1p(b)
      d = smith.sign(a)
      e = d * c
      return e
)JIT";

smith::jit::Module getSignedLog1pModel() {
  smith::jit::Module module("signed_log1p");
  module.define(signed_log1p_model);
  return module;
}
