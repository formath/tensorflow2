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

ids = tf.constant([[18287374, 3847113], [7174746, 18287374], [5173648, 5173648]])
embedding = embedding_ops.embedding_lookup_with_hash_table(emb_table,
                                                           ids,
                                                           is_training=True,
                                                           count_table=count_table,
                                                           count_filter_thr=1)

with tf.Session() as sess:
  sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
  emb = sess.run([embedding])
  print("embedding: %s" % (emb,))
