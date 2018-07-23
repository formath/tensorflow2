/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
class LegacyVar : public ResourceBase {
 public:
  explicit LegacyVar(DataType dtype) : tensor_(dtype) {}
  // Not copyable or movable.
  LegacyVar(const LegacyVar&) = delete;
  LegacyVar& operator=(const LegacyVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tensor_;

  ~LegacyVar() override {}
};

RemoteVariableOp::RemoteVariableOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  dtype_ = RemoveRefType(context->output_type(0));
}

void RemoteVariableOp::Compute(OpKernelContext* ctx) {
  /*
  mutex_lock l(init_mu_);
  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true));
    initialized_ = true;
  }
  auto creator = [this](LegacyVar** var) {
    *var = new LegacyVar(dtype_);
    (*var)->tensor()->set_shape(shape_);
    return Status::OK();
  };
  LegacyVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<LegacyVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));

  // 填充tensor
  Buffer<float>* buf = new Buffer<float>(cpu_allocator(), var->tensor()->NumElements());
  char* data = buf->template base<char>();
  if (data == nullptr) {
    buf->Unref();
    OP_REQUIRES(
        ctx, data != nullptr,
        errors::InvalidArgument("No more memory for RemoteVariableOp"));
  }
  std::vector<float> temp(var->tensor()->NumElements(), 0.5);
  memcpy(data, temp.data(), temp.size() * sizeof(float));
  var->tensor()->buf_ = buf;

  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.
  ctx->set_output_ref(0, var->mu(), var->tensor());
  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
  */

  Tensor* output;
  ctx->allocate_output(0, shape_, &output);
  auto output_flat = output->flat<float>();
  std::vector<float> temp(var->tensor()->NumElements(), 0.5);
  memcpy(output_flat.data(), temp.data(), temp.size() * sizeof(float));
}

}  // namespace tensorflow