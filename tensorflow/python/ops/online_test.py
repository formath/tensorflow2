import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.contrib.lookup import lookup_ops


sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]],
                         values=[18287374, 3847113, 7174746, 18287374, 5173648, 5173648],
                         dense_shape=[4, 1000])
sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]],
                             values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             dense_shape=[4, 1000])
embedding = tf.nn.embedding_lookup_sparse_v2(sp_ids,
                                             sp_weights=sp_weights,
                                             embedding_size=8,
                                             name="embedding_lookup_sparse_v2",
                                             combiner="sum")

weight = tf.get_variable("weight", initializer=tf.random_normal([8, 2], dtype=tf.float32))
logit = tf.matmul(embedding, weight)
label = ops.convert_to_tensor([1, 0, 1, 0], dtype=tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
loss = tf.reduce_mean(cross_entropy)

opt = tf.train.AdagradOptimizer(learning_rate=0.01)
grads = tf.gradients(loss, [embedding])
train = opt.minimize(loss)

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    tf.train.write_graph(sess.graph.as_graph_def(), './', 'graph.txt', as_text=True)
#    writer = tf.summary.FileWriter(logdir='./logdir', graph=sess.graph)
#    writer.flush()
#    exit()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

i = 0
while i < 1:
  sess.run(train)
  print("loss = %d" % loss.eval(session=sess))
  print("embedding = %s" % embedding.eval(session=sess))
  #print("grads = %s" % grads[0].eval(session=sess))
  i += 1

