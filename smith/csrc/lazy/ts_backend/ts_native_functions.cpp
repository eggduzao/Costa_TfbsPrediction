#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Functions.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <smith/csrc/lazy/core/helpers.h>
#include <smith/csrc/lazy/core/ir_builder.h>
#include <smith/csrc/lazy/core/metrics.h>
#include <smith/csrc/lazy/core/ops/utils.h>
#include <smith/csrc/lazy/core/shape_inference.h>
#include <smith/csrc/lazy/core/tensor_impl.h>
#include <smith/csrc/lazy/core/tensor_util.h>
#include <smith/csrc/lazy/generated/LazyNativeFunctions.h>
#include <smith/csrc/lazy/ts_backend/config.h>
#include <smith/csrc/lazy/ts_backend/ops/to_copy.h>
#include <smith/csrc/lazy/ts_backend/tensor_aten_ops.h>
#include <smith/csrc/lazy/ts_backend/ts_autograd_functions.h>
#include <smith/csrc/lazy/ts_backend/ts_eager_fallback.h>
#include <smith/library.h>

#include <utility>

using at::Tensor;

namespace smith::lazy {
namespace {

at::Tensor CreateLtcTensor(
    const at::Tensor& tensor,
    const std::optional<smith::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    return smith::lazy::CreateAtenFromLtcTensor(
        smith::lazy::LazyTensor::Create(tensor, *device));
  }
  return tensor;
}

std::optional<smith::lazy::BackendDevice> GetLtcDevice(
    const std::optional<c10::Device>& device) {
  if (!device) {
    return std::nullopt;
  }
  if (device->type() != at::kLazy) {
    return std::nullopt;
  }
  return smith::lazy::atenDeviceToBackendDevice(*device);
}

} // namespace

// clone is special in LT because we make it a no-op.
// This should be safe to do, because every operator in the LT is functional.
at::Tensor LazyNativeFunctions::clone(
    const at::Tensor& self,
    std::optional<at::MemoryFormat> memory_format) {
  auto self_lt = smith::lazy::TryGetLtcTensor(self);
  return smith::lazy::CreateAtenFromLtcTensor(
      self_lt->Create(self_lt->GetIrValue(), self_lt->GetDevice()));
}

at::Tensor LazyNativeFunctions::_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  SMITH_LAZY_FN_COUNTER("lazy::");
  auto dst_tensor = smith::lazy::TryGetLtcTensor(dst);
  auto self_tensor = smith::lazy::TryGetLtcTensor(self);
  if (!self_tensor) {
    // providing a new 'eager' value (self) for an existing lazy tensor (dst)
    static bool sync_update = FLAGS_smith_lazy_ts_tensor_update_sync;
    SMITH_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    // materializing a lazy tensor (self) and copying its value into eager
    // tensor (dst) detached=false lets us skip a copy in `ToTensor`, which
    // should be safe because we are only going to use the tensor for
    // dst.copy_()
    SMITH_CHECK(self_tensor);
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/false);
    at::Tensor typed_tensor =
        smith::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // Copying one lazy tensor to another
    if (!dst_tensor->CurrentIrValue()) {
      // if dest is not backed by IR (e.g. result of some lazy operation),
      // then it should have at::Tensor data backing it instead
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      SMITH_CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        // both src/dst are simply backed by at::Tensor data, no IR- do a
        // straightforward copy
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        // src needs to be materialized before its result can be used for a copy
        // into dst since we use the src tensor only for making a copy, we don't
        // need to detach it note: it would be even more efficient if we could
        // cause ToTensor to materialize the value directly into dst's buffer
        // (that would need to be detached though).
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/false));
      }
    } else {
      copy_(dst_tensor, self_tensor);
      auto* impl =
          dynamic_cast<smith::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
      impl->set_tensor(dst_tensor);
    }
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  SMITH_LAZY_FN_COUNTER("lazy::");
  auto dst_tensor = smith::lazy::TryGetLtcTensor(dst);
  auto self_tensor = smith::lazy::TryGetLtcTensor(self);
  if (!self_tensor) {
    SMITH_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    SMITH_CHECK(self_tensor);
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        smith::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is a lazy tensor
    auto* dest_impl =
        dynamic_cast<smith::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    SMITH_CHECK(dest_impl);
    dest_impl->tensor()->UpdateFromTensorOut(self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_to_copy(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  if (force_eager_fallback(at::aten::_to_copy)) {
    SMITH_INTERNAL_ASSERT(
        false,
        "Fallback is currently impossible for _to_copy since the fallback helper itself reinvokes _to_copy");
  }

  auto options = self.options();
  if (dtype) {
    // I put each of these setters in a conditional instead of doing
    // `self.options().dtype(dtype).layout(layout)... because calling
    // .dtype(nullopt) on an options() that already has dtype appears to wipe it
    options = options.dtype(dtype);
  }
  if (layout) {
    options = options.layout(layout);
  }
  if (memory_format) {
    options = options.memory_format(memory_format);
  }
  if (pin_memory) {
    // TODO(whc) can we honor 'pin_memory' in some/all cases?
    options = options.pinned_memory(pin_memory);
    SMITH_WARN_ONCE(
        "Pinned memory used in lazy _to_copy, check if the behavior is as intended");
  }

  SMITH_LAZY_FN_COUNTER("lazy::");
  auto lazy_self = smith::lazy::TryGetLtcTensor(self);
  if (!lazy_self && device && device->type() == c10::kLazy) {
    // Case 1: eager->lazy (we create a new lazy tensor)
    // See Note [Lazy Tensor Functionalization]
    // Invariant: if the functionalization key is in the exclude set, then we're
    // expected to return an ordinary tensor, which will be "lifted" into a
    // functional wrapper later.
    bool functionalize_output =
        !c10::impl::tls_local_dispatch_key_set().excluded_.has(
            c10::DispatchKey::Functionalize);
    return smith::lazy::to_lazy_tensor(
        self,
        options,
        *device,
        /*non_blocking=*/non_blocking,
        /*functionalize_output=*/functionalize_output);
  } else if (device && device->type() != c10::kLazy) {
    // Case 2: lazy->eager (forces a graph break since we are materializing a
    // tensor)

    SMITH_INTERNAL_ASSERT(lazy_self);
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    options = options.device(device);
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
    return moved_eager_tensor;
  } else if (
      device && device->type() == c10::kLazy && device->has_index() &&
      device->index() != self.device().index()) {
    // Case 3: lazy:0 -> lazy:1

    // TODO(whc) what do we actually want to do here?
    //   option 1: materialize, move eager tensor, create new lazy tensor
    //     - this should be our default, as it is what would happen before we
    //     implemented _to_copy
    //     - actually combines case 1 + case 2
    //   option 2: support multiple devices inside one lazy/TS executor (case 4)
    //     - but: we may have other assumptions that there is just one device
    //     per executor? so don't take this lightly

    SMITH_INTERNAL_ASSERT(lazy_self);
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    // we move the eager tensor to the 'eager' equivalent of our lazy device
    // e.g. if our device is lazy:1, the backend maps that to cuda:1, which is
    // what we use
    auto eager_device = c10::Device(
        smith::lazy::getBackend()->EagerFallbackDeviceType(), device->index());
    options = options.device(eager_device);
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/false, /*copy=*/true);
    lazy_self = smith::lazy::GetOrCreateLtcTensor(
        moved_eager_tensor,
        smith::lazy::atenDeviceToBackendDevice(eager_device));
    return smith::lazy::CreateAtenFromLtcTensor(lazy_self);

  } else {
    // Case 4: lazy->lazy (special case: keep the _to_copy INSIDE the lazy
    // graph)

    // Note: captured _to_copy will be executed with real eager tensors, not
    // lazy tensors. We DO NOT want to burn 'lazy:0' as the device into this
    // captured IR, or we will try to convert an eager tensor back to a lazy one
    // inside the smithscript executor lazy:0 -> lazy:1 is handled in case3, so
    // we can safely drop the device argument
    device = std::nullopt;

    smith::lazy::NodePtr node = smith::lazy::ReuseNode<ToCopy>(
        lazy_self->GetIrValue(),
        dtype,
        layout,
        device,
        pin_memory,
        non_blocking,
        memory_format);
    if (!node) {
      auto shapes = smith::lazy::compute_shape__to_copy(
          self, dtype, layout, device, pin_memory, non_blocking, memory_format);
      SMITH_INTERNAL_ASSERT(shapes.size() == 1);
      node = smith::lazy::MakeNode<ToCopy>(
          lazy_self->GetIrValue(),
          dtype,
          layout,
          device,
          pin_memory,
          non_blocking,
          memory_format,
          std::move(shapes));
      CacheNode(node);
    }

    auto result =
        smith::lazy::CreateAtenFromLtcTensor(smith::lazy::LazyTensor::Create(
            std::move(node), lazy_self->GetDevice()));
    return result;
  }
}

at::Tensor LazyNativeFunctions::empty_symint(
    at::SymIntArrayRef sym_size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  // TODO: support this directly
  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);
  const auto device_type = smith::lazy::getBackend()->EagerFallbackDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  auto tensor = CreateLtcTensor(x_result, GetLtcDevice(device));
  // See Note [Lazy Tensor Functionalization]
  if (c10::impl::tls_local_dispatch_key_set().excluded_.has(
          c10::DispatchKey::Functionalize)) {
    // Invariant: if the functionalization key is in the exclude set, then we're
    // expected to return an ordinary tensor, which will be "lifted" into a
    // functional wrapper later.
    return tensor;
  } else {
    auto wrapped = at::functionalization::impl::to_functional_tensor(tensor);
    return wrapped;
  }
}

at::Tensor LazyNativeFunctions::empty_strided_symint(
    at::SymIntArrayRef sym_size,
    at::SymIntArrayRef sym_stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  SMITH_LAZY_FN_COUNTER("lazy::");
  at::Tensor t =
      empty_symint(sym_size, dtype, layout, device, pin_memory, std::nullopt);
  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);
  auto stride = C10_AS_INTARRAYREF_SLOW(sym_stride);
  return t.as_strided(size, stride, /*storage_offset=*/0);
}

at::Tensor& LazyNativeFunctions::fill_(
    at::Tensor& self,
    const at::Scalar& value) {
  SMITH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = smith::lazy::TryGetLtcTensor(self);
  smith::lazy::fill_(self_tensor, value);
  return self;
}

at::Tensor LazyNativeFunctions::max_pool3d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  return smith::lazy::MaxPool3dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

// We need to explicitly override max pooling operators and just call the
// fallback for them because we've customized the autograd function for them
// (backward needs saved indices from forward).
std::tuple<at::Tensor, at::Tensor> LazyNativeFunctions::max_pool3d_with_indices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  return at::native::
      call_fallback_fn<&ltc_eager_fallback, ATEN_OP(max_pool3d_with_indices)>::
          call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor LazyNativeFunctions::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  return at::native::call_fallback_fn<
      &ltc_eager_fallback,
      ATEN_OP(max_pool3d_with_indices_backward)>::
      call(
          grad_output,
          self,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices);
}

at::Tensor LazyNativeFunctions::_unsafe_view(
    const at::Tensor& self,
    at::IntArrayRef size) {
  SMITH_LAZY_FN_COUNTER("lazy::");
  return LazyNativeFunctions::view_copy_symint(
      self, c10::fromIntArrayRefSlow(size));
}

// This is needed by the smith.tensor constructor.
// LazyTensor always opts into functionalization.
// "lifting" a tensor for functionalization means wrapping it in a
// FunctionalTensorWrapper object.
at::Tensor LazyNativeFunctions::lift(const at::Tensor& tensor) {
  SMITH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  return at::functionalization::impl::to_functional_tensor(tensor);
}
at::Tensor LazyNativeFunctions::lift_fresh(const at::Tensor& tensor) {
  SMITH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  return at::functionalization::impl::to_functional_tensor(tensor);
}

// All of the below ops correspond to CompositeExplicitAutograd kernels from
// core that call into view operators internally. These are all composite ops
// that LTC can technically reuse / get for free, but we need to
// "functionalize" them to remove the view ops before we can use them.
at::Tensor LazyNativeFunctions::block_diag(at::TensorList tensors) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      block_diag)>::call(tensors);
}
at::Tensor LazyNativeFunctions::new_empty_strided_symint(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return at::functionalization::
      functionalize_aten_op_symint<ATEN_OP(new_empty_strided)>::call(
          self, size, stride, dtype, layout, device, pin_memory);
}

at::Tensor LazyNativeFunctions::narrow_copy_symint(
    const at::Tensor& self,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt length) {
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      narrow_copy)>::call(self, dim, std::move(start), std::move(length));
}
at::Tensor LazyNativeFunctions::pixel_shuffle(
    const at::Tensor& self,
    int64_t upscale_factor) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      pixel_shuffle)>::call(self, upscale_factor);
}
at::Tensor LazyNativeFunctions::pixel_unshuffle(
    const at::Tensor& self,
    int64_t downscale_factor) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      pixel_unshuffle)>::call(self, downscale_factor);
}
at::Tensor LazyNativeFunctions::select_backward_symint(
    const at::Tensor& grad_output,
    c10::SymIntArrayRef input_sizes,
    int64_t dim,
    c10::SymInt index) {
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      select_backward)>::call(grad_output, input_sizes, dim, std::move(index));
}
at::Tensor LazyNativeFunctions::_trilinear(
    const at::Tensor& i1,
    const at::Tensor& i2,
    const at::Tensor& i3,
    at::IntArrayRef expand1,
    at::IntArrayRef expand2,
    at::IntArrayRef expand3,
    at::IntArrayRef sumdim,
    int64_t unroll_dim) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(_trilinear)>::
      call(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}
at::Tensor LazyNativeFunctions::linalg_pinv(
    const at::Tensor& self,
    const std::optional<at::Tensor>& atol,
    const std::optional<at::Tensor>& rtol,
    bool hermitian) {
  return at::functionalization::functionalize_aten_op<ATEN_OP2(
      linalg_pinv, atol_rtol_tensor)>::call(self, atol, rtol, hermitian);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> LazyNativeFunctions::svd(
    const at::Tensor& self,
    bool some,
    bool compute_uv) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(svd)>::call(
      self, some, compute_uv);
}

// functionalize_aten_op can't handle out= ops directly.
// Instead, we can call the composite kernel from core, and copy and mutations
// back to the inputs.
at::Tensor& LazyNativeFunctions::logsumexp_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out) {
  auto self_wrapped = at::functionalization::impl::to_functional_tensor(self);
  auto out_wrapped = at::functionalization::impl::to_functional_tensor(out);
  // directly call the composite kernel from core.
  // Make sure to re-enable functionalization first.
  auto curr_tls = c10::impl::tls_local_dispatch_key_set();
  auto tls_reenable_functionalize = c10::impl::PODLocalDispatchKeySet();
  tls_reenable_functionalize.set_included(curr_tls.included_);
  tls_reenable_functionalize.set_excluded(
      curr_tls.excluded_.remove(c10::DispatchKey::Functionalize));
  c10::impl::ForceDispatchKeyGuard guard_(tls_reenable_functionalize);
  at::native::logsumexp_out(self_wrapped, dim, keepdim, out_wrapped);
  auto out_unwrapped =
      at::functionalization::impl::from_functional_tensor(out_wrapped);
  // propagate mutations back to the inputs (including resizing)
  out.resize_(out_unwrapped.sizes());
  out.copy_(out_unwrapped);
  return out;
}

at::Tensor LazyNativeFunctions::diag_embed(
    const at::Tensor& self,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      diag_embed)>::call(self, offset, dim1, dim2);
}

at::Tensor LazyNativeFunctions::diagonal_backward_symint(
    const at::Tensor& grad_output,
    at::SymIntArrayRef input_sizes,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      diagonal_backward)>::call(grad_output, input_sizes, offset, dim1, dim2);
}

at::Tensor LazyNativeFunctions::slice_backward_symint(
    const at::Tensor& grad_output,
    at::SymIntArrayRef input_sizes,
    int64_t dim,
    c10::SymInt start,
    c10::SymInt end,
    c10::SymInt step) {
  return at::functionalization::
      functionalize_aten_op_symint<ATEN_OP(slice_backward)>::call(
          grad_output,
          input_sizes,
          dim,
          std::move(start),
          std::move(end),
          std::move(step));
}

// reuse the composite kernel from core, that way we don't need to provide a
// backwards formula for native_group_norm
std::tuple<Tensor, Tensor, Tensor> LazyNativeFunctions::native_group_norm(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  return at::native::math_group_norm(
      input, weight, bias, N, C, HxW, group, eps);
}

} // namespace smith::lazy
