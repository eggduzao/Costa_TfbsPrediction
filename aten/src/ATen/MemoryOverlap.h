#pragma once

#include <c10/macros/Export.h>

namespace c10 {
struct TensorImpl;
}

namespace at {
class TensorBase;

// MemOverlap: Whether or not there is memory overlap
//
// No: Absolutely no memory overlap
// Yes: Absolutely yes memory overlap
// TooHard: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
enum class MemOverlap { No, Yes, TooHard };

enum class MemOverlapStatus { Full, Partial, No, TooHard };

SMITH_API MemOverlap has_internal_overlap(const TensorBase& t);
SMITH_API MemOverlap has_internal_overlap(c10::TensorImpl* t);

SMITH_API void assert_no_internal_overlap(const TensorBase& t);
SMITH_API void assert_no_internal_overlap(c10::TensorImpl* t);

SMITH_API MemOverlapStatus
get_overlap_status(const TensorBase& a, const TensorBase& b);
SMITH_API MemOverlapStatus
get_overlap_status(const c10::TensorImpl* a, const c10::TensorImpl* b);

SMITH_API void assert_no_partial_overlap(
    const TensorBase& a,
    const TensorBase& b);
void assert_no_partial_overlap(c10::TensorImpl* a, c10::TensorImpl* b);

SMITH_API void assert_no_overlap(const TensorBase& a, const TensorBase& b);
SMITH_API void assert_no_overlap(c10::TensorImpl* a, c10::TensorImpl* b);

} // namespace at
