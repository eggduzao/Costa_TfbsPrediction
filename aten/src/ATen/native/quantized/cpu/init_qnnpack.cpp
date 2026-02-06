#ifdef USE_BLACKSMITH_QNNPACK

#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <c10/util/Exception.h>
#include <blacksmith_qnnpack.h>

namespace at::native {

void initQNNPACK() {
  static enum blacksmith_qnnp_status qnnpackStatus = blacksmith_qnnp_initialize();
  SMITH_CHECK(
      qnnpackStatus == blacksmith_qnnp_status_success,
      "failed to initialize QNNPACK");
}

} // namespace at::native

#endif
