
import tensorflow as tf
from tensorflow.contrib import layers

def attention(queries,
              keys,
              values,
              num_units,
              num_output_units,
              activation_fn,
              normalizer_fn,
              normalizer_params,
              reuse,
              scope,
              variables_collections=None,
              outputs_collections=None,
              query_masks=None,
              key_masks=None,
              num_heads=8,
              need_linear_transform=True):

    s_item_vec, stt_vec = multihead_attention(queries=queries,
                                              keys=keys,
                                              values=values,
                                              num_units=num_units,
                                              num_output_units=num_output_units,
                                              activation_fn=None,
                                              normalizer_fn=normalizer_fn,
                                              normalizer_params=normalizer_params,
                                              scope=scope,
                                              reuse=reuse,
                                              query_masks=query_masks,
                                              key_masks=key_masks,
                                              variables_collections=variables_collections,
                                              outputs_collections=outputs_collections,
                                              num_heads=num_heads,
                                              need_linear_transform=need_linear_transform)

    item_vec = feedforward(s_item_vec,
                           num_units=[num_output_units * 2, num_output_units],
                           activation_fn=activation_fn,
                           normalizer_fn=normalizer_fn,
                           normalizer_params=normalizer_params,
                           scope=scope + "_feed_forward",
                           reuse=reuse,
                           variables_collections=variables_collections,
                           outputs_collections=outputs_collections)


    return item_vec


def multihead_attention(queries,
                        keys,
                        values,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        num_heads=8,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        variables_collections=None,
                        outputs_collections=None,
                        need_linear_transform=True):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    num_output_units: A scalar. Output Value size.
    keep_prob: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
    query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
    key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
        num_units = queries.get_shape().as_list()[-1]

    if num_output_units is None:
        num_output_units = keys.get_shape().as_list()[-1]
    if values is None:
        values = keys

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    if need_linear_transform:
        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = layers.fully_connected(queries,
                                   num_units,
                                   activation_fn=activation_fn,
                                   normalizer_fn=normalizer_fn,
                                   normalizer_params=normalizer_params,
                                   variables_collections=variables_collections,
                                   outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
        K = layers.fully_connected(keys,
                                   num_units,
                                   activation_fn=activation_fn,
                                   normalizer_fn=normalizer_fn,
                                   normalizer_params=normalizer_params,
                                   variables_collections=variables_collections,
                                   outputs_collections=outputs_collections, scope="K")  # (N, T_k, C)
        V = layers.fully_connected(values,
                                   num_output_units,
                                   activation_fn=activation_fn,
                                   normalizer_fn=normalizer_fn,
                                   normalizer_params=normalizer_params,
                                   variables_collections=variables_collections,
                                   outputs_collections=outputs_collections, scope="V")  # (N, T_k, C)
    # else:
    #     Q = layers.dropout(queries, keep_prob=1.0, is_training=normalizer_params['is_training'],outputs_collections=outputs_collections, scope="Q")
    #     K = layers.dropout(keys, keep_prob=1.0, is_training=normalizer_params['is_training'],outputs_collections=outputs_collections, scope="K")
    #     V = layers.dropout(values, keep_prob=1.0, is_training=normalizer_params['is_training'],outputs_collections=outputs_collections, scope="V")

    Q = queries
    K = keys
    V = values

    def split_last_dimension_then_transpose(tensor, num_heads):
      t_shape = tensor.get_shape().as_list()
      tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, t_shape[-1] // num_heads])
      return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, t_shape[-1]]

    Q_ = split_last_dimension_then_transpose(Q, num_heads)  # (h*N, T_q, C/h)
    K_ = split_last_dimension_then_transpose(K, num_heads)  # (h*N, T_k, C/h)
    V_ = split_last_dimension_then_transpose(V, num_heads)  # (h*N, T_k, C'/h)
    Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
    K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
    # [batch_size, num_heads, query_len, key_len]

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    query_len = queries.get_shape().as_list()[1]
    key_len = keys.get_shape().as_list()[1]

    key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, 1, key_len]),
                        [1, num_heads, query_len, 1])
    paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
    outputs = tf.where(key_masks, outputs, paddings)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    query_masks = tf.tile(tf.reshape(query_masks, [-1, 1, query_len, 1]),
                          [1, num_heads, 1, key_len])
    paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
    outputs = tf.where(query_masks, outputs, paddings)

    # Attention vector
    att_vec = outputs

    # Dropouts
    # outputs = layers.dropout(outputs, keep_prob=keep_prob, is_training=is_training)

    # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    def transpose_then_concat_last_two_dimenstion(tensor):
      tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
      t_shape = tensor.get_shape().as_list()
      num_heads, dim = t_shape[-2:]
      return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

    outputs = transpose_then_concat_last_two_dimenstion(outputs)  # (N, T_q, C)

    # Residual connection
    # outputs += queries
    # layers.batch_norm
    # Normalize
    # outputs = layers.layer_norm(outputs)  # (N, T_q, C)

  return outputs, att_vec


def feedforward(inputs,
                num_units=[2048, 512],
                activation_fn=None,
                normalizer_fn=None,
                normalizer_params=None,
                scope="feedforward",
                reuse=None,
                variables_collections=None,
                outputs_collections=None):
  '''Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    outputs = layers.fully_connected(inputs,
                                     num_units[0],
                                     activation_fn=activation_fn,
                                     normalizer_fn = normalizer_fn,
                                     normalizer_params = normalizer_params,
                                     variables_collections=variables_collections,
                                     outputs_collections=outputs_collections)
    outputs = layers.fully_connected(outputs,
                                     num_units[1],
                                     activation_fn=None,
                                     normalizer_fn=normalizer_fn,
                                     normalizer_params=normalizer_params,
                                     variables_collections=variables_collections,
                                     outputs_collections=outputs_collections)

    outputs += inputs
    outputs = layers.layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1)
  return outputs
