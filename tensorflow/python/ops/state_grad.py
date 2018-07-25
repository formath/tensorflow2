# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Gradients for operators defined in state_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops


# TODO(b/31222613): These ops may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable("Assign")


ops.NotDifferentiable("AssignAdd")


ops.NotDifferentiable("AssignSub")


ops.NotDifferentiable("ScatterAdd")


ops.NotDifferentiable("ScatterSub")


ops.NotDifferentiable("ScatterMul")


ops.NotDifferentiable("ScatterDiv")


ops.NotDifferentiable("ScatterNdUpdate")

ops.NotDifferentiable("ScatterNdAdd")

ops.NotDifferentiable("ScatterNdSub")

ops.NotDifferentiable("ScatterNdMul")

ops.NotDifferentiable("ScatterNdDiv")

@ops.RegisterGradient("RemoteSparseVariable")
def _RemoteSparseVariableGrad(op, grad):
  """Gradient for GatherV2 op."""
  # params can be large, so colocate the shape calculation with it.
  #
  # params can be very large for sparse model, array_ops.shape raises
  # exception on the Windows platform when any dimension is larger than
  # int32. params_shape is not used in optimizer apply_sparse gradients,
  # so it's fine to convert it back to int32 regardless of truncation.
  sparse_ids = op.inputs[0]
  grads = op.inputs[1]

  gen_state_ops.remote_sparse_variable_grad(keys=sparse_ids, grads=grads)
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    params_shape = math_ops.to_int32(params_shape)

  indices = op.inputs[1]
  indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
  axis = op.inputs[2]
  axis_static = tensor_util.constant_value(axis)

  # For axis 0 gathers, build an appropriately shaped IndexedSlices.
  if axis_static == 0:
    if context.executing_eagerly():
      params_tail_shape = params_shape.cpu()[1:]
    else:
      params_tail_shape = params_shape[1:]
    values_shape = array_ops.concat([indices_size, params_tail_shape], 0)
    values = array_ops.reshape(grad, values_shape)
    indices = array_ops.reshape(indices, indices_size)
    return [ops.IndexedSlices(values, indices, params_shape), None, None]

  outer_shape = params_shape[:axis]
  outer_dims = array_ops.size(outer_shape)
  inner_shape = params_shape[axis:][1:]
  inner_dims = array_ops.size(inner_shape)

  outer_axes_indices = math_ops.range(outer_dims)
  inner_axes_indices = math_ops.range(outer_dims + 1,
                                      outer_dims + 1 + inner_dims)

  values_shape = array_ops.concat([outer_shape, indices_size, inner_shape], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, indices_size)

  # We need to sum up every slice `values[..., i, ....]` corresponding to
  # `params[..., indices[i], ...]`. Since `unsorted_segment_sum` does not
  # support an axis parameter, we transpose the gather dimension to the front,
  # then use `unsorted_segment_sum` to build a
  # [gather_axis, outer_axes, inner_axes] tensor with all the gradients
  # affecting each index in `gather_axis` summed up.
  transpose_dims = array_ops.concat(
      [[outer_dims], outer_axes_indices, inner_axes_indices], 0)
  values_transpose = array_ops.transpose(values, transpose_dims)
  num_segments = params_shape[axis]

  params_grad = math_ops.unsorted_segment_sum(values_transpose, indices,
                                              num_segments)

  # Inverts the above transpose by moving dimension 0 back to its original
  # position.
  invert_transpose_dims = array_ops.concat(
      [outer_axes_indices + 1, [0], inner_axes_indices], 0)
  params_grad = array_ops.transpose(params_grad, invert_transpose_dims)
  return [params_grad, None, None]