# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import logging_ops
from tensorflow.python.training import slot_creator
from tensorflow.python.training import adam
from tensorflow.python.training import training_ops
from tensorflow.contrib.lookup import PartitionedMutableHashTable
from tensorflow.contrib.lookup import MutableHashTable
import re
SPARSE_ADAM_OPTIMIZER_MIN_DIM = 8


class HashAdamOptimizer(adam.AdamOptimizer):
  
  def _get_real_name(self, var_name):
    return re.sub(r'[/:]', '_', var_name)

  def _var_key(self, var):
    if hasattr(var, "op"):
      return (var.op.graph, var.op.name)
    if self._is_hash_table(var):
      return (var._table_ref.graph, var.name)
    return var._unique_id

  def _hash_slot(self, var, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if self._var_key(var) not in named_slots:
      with ops.name_scope("hash_slot_%s" % slot_name):
        if self._is_hash_table(var):
          default_value = array_ops.zeros(array_ops.size(var._default_value))
        else:
          default_value = array_ops.zeros(array_ops.size(var.shape[1]))
        shared_name = "%s_%s" % (op_name, slot_name)
        table_name = "hash_table_%s_%s_%s" % (op_name,
                                              slot_name,
                                              self._get_real_name(var.name))
        if self._is_hash_table(var):
          dtype = var.value_dtype.base_dtype
        else:
          dtype = var.dtype.base_dtype
        new_slot_variable = MutableHashTable(
            dtypes.int64, dtype, default_value, shared_name, table_name)
        named_slots[self._var_key(var)] = new_slot_variable
    return named_slots[self._var_key(var)]

  def _zeros_slot(self, var, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if self._var_key(var) not in named_slots:
      with ops.name_scope("zeros_slot_%s" % slot_name):
          new_slot_variable = slot_creator.create_zeros_slot(
              var, op_name)
          named_slots[self._var_key(var)] = new_slot_variable
    return named_slots[self._var_key(var)]

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    if self._is_hash_table(first_var):
      self._create_non_slot_variable(initial_value=self._beta1,
                                      name="beta1_power",
                                      colocate_with=first_var._table_ref)
      self._create_non_slot_variable(initial_value=self._beta2,
                                      name="beta2_power",
                                      colocate_with=first_var._table_ref)
    else:
      self._create_non_slot_variable(initial_value=self._beta1,
                                      name="beta1_power",
                                      colocate_with=first_var)
      self._create_non_slot_variable(initial_value=self._beta2,
                                      name="beta2_power",
                                      colocate_with=first_var)
    # Create slots for the first and second moments.
    for v in var_list:
      if self._is_hash_table(v):
        self._hash_slot(v, "m", self._name)
        self._hash_slot(v, "v", self._name)
      else:
        if len(v.get_shape()) != 2 or v.get_shape()[0] < SPARSE_ADAM_OPTIMIZER_MIN_DIM:
          self._zeros_slot(v, "m", self._name)
          self._zeros_slot(v, "v", self._name)
        else:
          self._hash_slot(v, "m", self._name)
          self._hash_slot(v, "v", self._name)

  def _cast_hparams(self, var):
    if self._is_hash_table(var):
      dtype = var.value_dtype.base_dtype
    else:
      dtype = var.dtype.base_dtype
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, dtype)
    beta2_power = math_ops.cast(beta2_power, dtype)
    lr_t = math_ops.cast(self._lr_t, dtype)
    beta1_t = math_ops.cast(self._beta1_t, dtype)
    beta2_t = math_ops.cast(self._beta2_t, dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, dtype)
    return beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t
      

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    if isinstance(m, MutableHashTable):
      self._zeros_slot(var, "m_dense", self._name)
      self._zeros_slot(var, "v_dense", self._name)
      m = self.get_slot(var, "m_dense")
      v = self.get_slot(var, "v_dense")
    else:
      v = self.get_slot(var, "v")

    beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t = self._cast_hparams(
        var)
    return training_ops.apply_adam(
      var, m, v,
      beta1_power,
      beta2_power,
      lr_t,
      beta1_t,
      beta2_t,
      epsilon_t,
      grad,
      use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):
    return control_flow_ops.cond(array_ops.size(grad.indices) > 0,
      lambda: self._apply_sparse_internal(grad, var), lambda: gen_control_flow_ops.no_op())

  # var 是一个embedding的矩阵，第一维度是字典的size，第二维度是嵌入向量的维度
  # grad 是一个稀疏的梯度，indeces是指的var中的行号，values是指这行的梯度
  def _apply_sparse_internal(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    #如果var是一个hash table，则slot一定也是一个hash table，走 _apply_to_hash_table 的逻辑
    if self._is_hash_table(var):
      return self._apply_to_hash_table(grad, var, m, v)
    #如果var不是一个hash table，则slot分两种情况，一种是hash table，一种是普通的variable
    elif isinstance(m, MutableHashTable):
      return self._apply_to_variable_with_hash_slot(grad, var, m, v)
    else:
      return self._apply_to_variable_with_variable_slot(grad, var, m, v)

  def _apply_to_hash_table(self, grad, var, m, v):
    beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t = self._cast_hparams(
        var)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    m_t, v_t = self._read_and_update_hash_table_slot(grad, m, v, beta1_t, beta2_t, epsilon_t)
    denominator_slice = math_ops.sqrt(v_t) + epsilon_t
    var_old_value = var.lookup(grad.indices, "hash_find_%s" % var.name)
    g = (lr * m_t / denominator_slice)
    #g = logging_ops.Print(g,[var_old_value,g],"梯度")
    var_new_value = var_old_value - g
    with ops.control_dependencies([var.insert(grad.indices,var_new_value)]):
      #var._table_ref = logging_ops.Print(var._table_ref,var.export(),"var hashTable存的值")
      return control_flow_ops.group(var._table_ref, m_t, v_t)


  def _apply_to_variable_with_hash_slot(self, grad, var, m, v):
    beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t = self._cast_hparams(
        var)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    m_t, v_t = self._read_and_update_hash_table_slot(grad, m, v, beta1_t, beta2_t, epsilon_t)
    denominator_slice = math_ops.sqrt(v_t) + epsilon_t
    var_update = state_ops.scatter_sub(var, grad.indices,
                                        lr * m_t / denominator_slice,
                                        use_locking=self._use_locking)
    return control_flow_ops.group(var_update, m_t, v_t)

  def _apply_to_variable_with_variable_slot(self, grad, var, m, v):
    beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t = self._cast_hparams(
        var)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    m_t = state_ops.scatter_update(m, grad.indices,
                                    beta1_t * array_ops.gather(m, grad.indices) +
                                    (1 - beta1_t) * grad.values,
                                    use_locking=self._use_locking)
    m_t_slice = array_ops.gather(m_t, grad.indices)
    v_t = state_ops.scatter_update(v, grad.indices,
                                    beta2_t * array_ops.gather(v, grad.indices) +
                                    (1 - beta2_t) *
                                    math_ops.square(grad.values),
                                    use_locking=self._use_locking)

    m_t_slice = array_ops.gather(m_t, grad.indices)
    v_t_slice = array_ops.gather(v_t, grad.indices)
    denominator_slice = math_ops.sqrt(v_t_slice) + epsilon_t
    var_update = state_ops.scatter_sub(var, grad.indices,
                                        lr * m_t_slice / denominator_slice,
                                        use_locking=self._use_locking)
    return control_flow_ops.group(var_update, m_t, v_t)

  def _read_and_update_hash_table_slot(self, grad, m, v, beta1_t, beta2_t, epsilon_t):
    # \\(m := beta1 * m + (1 - beta1) * g_t\\)
    # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
    # \\(variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
    mv_keys = math_ops.cast(grad.indices, dtypes.int64)
    # m_m,v_m: 是指从m,v中查出mv_keys个m值的集合，第一维是mv_keys的个数，第二维是embedding_size
    #mv_keys = logging_ops.Print(mv_keys,[array_ops.shape(mv_keys)],"shape")
    m_m = m.lookup(mv_keys, "hash_find_%s" % m.name)
    v_m = v.lookup(mv_keys, "hash_find_%s" % v.name)
    m_scaled_g_values = grad.values * (1 - beta1_t)
    v_scaled_g_values = math_ops.square(grad.values) * (1 - beta2_t)
    m_t = m_m * beta1_t + m_scaled_g_values
    v_t = v_m * beta2_t + v_scaled_g_values
    with ops.control_dependencies([m.insert(mv_keys, m_t), v.insert(mv_keys, v_t)]):
      return m_t, v_t
