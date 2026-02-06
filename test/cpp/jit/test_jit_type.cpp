#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <smith/csrc/jit/testing/file_check.h>
#include "smith/csrc/jit/ir/ir.h"
#include "smith/csrc/jit/ir/irparser.h"

namespace smith {
namespace jit {

TEST(JitTypeTest, IsComplete) {
  auto tt = c10::TensorType::create(
      at::kFloat,
      at::kCPU,
      c10::SymbolicShape(std::vector<std::optional<int64_t>>({1, 49})),
      std::vector<c10::Stride>(
          {c10::Stride{2, true, 1},
           c10::Stride{1, true, 1},
           c10::Stride{0, true, std::nullopt}}),
      false);
  SMITH_INTERNAL_ASSERT(!tt->isComplete());
  SMITH_INTERNAL_ASSERT(!tt->strides().isComplete());
}

TEST(JitTypeTest, UnifyTypes) {
  auto bool_tensor = TensorType::get()->withScalarType(at::kBool);
  auto opt_bool_tensor = OptionalType::create(bool_tensor);
  auto unified_opt_bool = unifyTypes(bool_tensor, opt_bool_tensor);
  SMITH_INTERNAL_ASSERT(opt_bool_tensor->isSubtypeOf(**unified_opt_bool));

  auto tensor = TensorType::get();
  SMITH_INTERNAL_ASSERT(!tensor->isSubtypeOf(*opt_bool_tensor));
  auto unified = unifyTypes(opt_bool_tensor, tensor);
  SMITH_INTERNAL_ASSERT(unified);
  auto elem = (*unified)->expectRef<OptionalType>().getElementType();
  SMITH_INTERNAL_ASSERT(elem->isSubtypeOf(*TensorType::get()));

  auto opt_tuple_none_int = OptionalType::create(
      TupleType::create({NoneType::get(), IntType::get()}));
  auto tuple_int_none = TupleType::create({IntType::get(), NoneType::get()});
  auto out = unifyTypes(opt_tuple_none_int, tuple_int_none);
  SMITH_INTERNAL_ASSERT(out);

  std::stringstream ss;
  ss << (*out)->annotation_str();
  testing::FileCheck()
      .check("Optional[Tuple[Optional[int], Optional[int]]]")
      ->run(ss.str());

  auto fut_1 = FutureType::create(IntType::get());
  auto fut_2 = FutureType::create(NoneType::get());
  auto fut_out = unifyTypes(fut_1, fut_2);
  SMITH_INTERNAL_ASSERT(fut_out);
  SMITH_INTERNAL_ASSERT((*fut_out)->isSubtypeOf(
      *FutureType::create(OptionalType::create(IntType::get()))));

  auto dict_1 = DictType::create(IntType::get(), NoneType::get());
  auto dict_2 = DictType::create(IntType::get(), IntType::get());
  auto dict_out = unifyTypes(dict_1, dict_2);
  SMITH_INTERNAL_ASSERT(!dict_out);
}

} // namespace jit
} // namespace smith
