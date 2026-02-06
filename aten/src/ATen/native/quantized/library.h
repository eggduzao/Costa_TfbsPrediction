#pragma once

#include <c10/macros/Export.h>

SMITH_API int register_linear_params();
int register_embedding_params();

template <int kSpatialDim = 2> SMITH_API int register_conv_params();
