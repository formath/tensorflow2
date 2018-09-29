import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.contrib.lookup import lookup_ops

init_op_list = []

emb_table = lookup_ops.PartitionedMutableHashTable(tf.int64,
                                                   tf.float32,
                                                   [0.0, 0.0, 0.0],
                                                   shard_num=2,
                                                   name="sparse_id_embedding",
                                                   checkpoint=True,
                                                   trainable=True)
#emb_table = lookup_ops.MutableHashTable(tf.int64,
#                            tf.float32,
#                            [0.0, 0.0, 0.0],
#                            name="sparse_id_embedding",
#                            checkpoint=True)
#tf.add_to_collections([ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES], emb_table)

count_table = lookup_ops.PartitionedMutableHashTable(tf.int64,
                                                     tf.int64,
                                                     0,
                                                     shard_num=2,
                                                     name="sparse_id_counter",
                                                     checkpoint=True,
                                                     trainable=False)

sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]],
                         values=[18287374, 3847113, 7174746, 18287374, 5173648, 5173648],
                         dense_shape=[4, 1000])
sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]],
                             values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             dense_shape=[4, 1000])
embedding = embedding_ops.embedding_lookup_sparse_with_hash_table(emb_table,
                                                                  sp_ids,
                                                                  sp_weights,
                                                                  combiner="sum",
                                                                  is_training=True,
                                                                  count_table=count_table,
                                                                  count_filter_thr=1)
weight = tf.get_variable("weight", initializer=tf.random_normal([3, 2], mean=0.0, stddev=0.1, dtype=tf.float32))
logit = tf.matmul(embedding, weight)
label = ops.convert_to_tensor([1, 0, 1, 0], dtype=tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
loss = tf.reduce_mean(cross_entropy)

opt = tf.train.AdamOptimizer(learning_rate=0.01)
train = opt.minimize(loss)

with tf.Session() as sess:
  sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
  i = 0
  while i < 30:
    _, lo, emb, wht = sess.run([train, loss, embedding, weight])
    print("iter: %d loss: %s embedding: %s weight: %s" % (i, lo, emb, wht))
    i += 1
  tf.train.write_graph(sess.graph.as_graph_def(), './', 'graph.txt', as_text=True)
  writer = tf.summary.FileWriter(logdir='./logdir', graph=sess.graph)
  writer.flush()
