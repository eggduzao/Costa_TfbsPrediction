#include <vector>

#include <smith/csrc/lazy/core/tensor_util.h>
#include <smith/csrc/lazy/core/util.h>

namespace smith::lazy {

SMITH_API bool StrideIsSupported(c10::ArrayRef<int64_t> stride);

SMITH_API std::vector<int64_t> GetArrayStridePermutation(
    c10::ArrayRef<int64_t> stride);

SMITH_API Shape MakeDiagonalShape(
    const Shape& shape,
    int64_t offset,
    int64_t dim1,
    int64_t dim2);

SMITH_API Shape
MakePermuteShape(const Shape& source_shape, c10::ArrayRef<int64_t> permutation);

SMITH_API Shape MakeSelectShape(
    const Shape& shape,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride);

SMITH_API int64_t GetStride(int64_t start, int64_t end, int64_t stride);

SMITH_API std::vector<int64_t> BuildSqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim);

SMITH_API std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim);

} // namespace smith::lazy
