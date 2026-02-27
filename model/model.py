#!/usr/bin/env python
#-*- coding:utf8 -*-
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.ops import metrics
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from prada_model_ops.metrics import auc
from prada_interface.algorithm import Algorithm
from prada.runner.prada_exceptions import SkipNanInfException

from model_util.fg import FgParser
from model_util.util import *
import model_util.global_var as gl
from model_util.attention import attention
from optimizer.adagrad_decay import SearchAdagradDecay
from optimizer.adagrad import SearchAdagrad
from optimizer import optimizer_ops as myopt
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
from requests.exceptions import ConnectionError
import numpy as np
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn, _RealValuedColumn

np.set_printoptions(threshold=np.inf)

optimizer_dict = {
    "AdagradDecay": lambda opt_conf, global_step: SearchAdagradDecay(opt_conf).get_optimizer(global_step),
    "Adagrad": lambda opt_conf, global_step: SearchAdagrad(opt_conf).get_optimizer(global_step)
}

class CTR(Algorithm):

    def init(self, context):
        self.context = context
        self.logger = self.context.get_logger()

        gl._init()
        gl.set_value('logger', self.logger)

        self.config = self.context.get_config()
        self.mode = self.config.get_job_config("mode")
        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']

        self.colunm_blocks_dict = dict()
        for k, v in self.config.get_all_job_config().items():
            if k == 'input_columns':
                self.model_name = list(v.keys())[0]
                v = v[self.model_name]
                for block_name, columns in v.items():
                    column_list = []
                    for column in columns:
                        column_list.append(column)
                    self.colunm_blocks_dict[block_name] = column_list

        if self.model_name is None:
            self.model_name = "CTR"

        self.user_column_blocks = []
        self.item_column_blocks = []
        self.item_query_column_blocks = []
        self.bias_column_blocks = []
        self.user_profile_column_blocks = []

        if self.algo_config.get('user_columns') is not None:
            arr_blocks = self.algo_config.get('user_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.user_column_blocks.append(block)
        else:
            raise RuntimeError("user_columns must be specified.")

        if self.algo_config.get("item_columns") is not None:
            arr_blocks = self.algo_config['item_columns'].split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.item_column_blocks.append(block)
        else:
            raise RuntimeError("item_columns must be specified.")

        if self.algo_config.get("item_query_columns") is not None:
            arr_blocks = self.algo_config['item_query_columns'].split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.item_query_column_blocks.append(block)
        else:
            raise RuntimeError("item_query_columns must be specified.")

        if self.algo_config.get('bias_columns') is not None:
            arr_blocks = self.algo_config.get('bias_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.bias_column_blocks.append(block)

        if self.algo_config.get("user_profile_columns") is not None:
            arr_blocks = self.algo_config['user_profile_columns'].split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.user_profile_column_blocks.append(block)

        self.seq_column_blocks = []
        self.seq_column_len = {}
        if self.algo_config.get('seq_column_blocks') is not None:
            arr_blocks = self.algo_config.get('seq_column_blocks').split(';', -1)
            for block in arr_blocks:
                arr = block.split(':', -1)
                if len(arr) < 2: continue
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]

        self.layer_dict = {}
        self.sequence_layer_dict = {}
        self.query_sequence_layer_dict = {}
        self.layer_net_dict = {}

        self.collections_dnn_hidden_layer = self.model_name + "_collections_dnn_hidden_layer"
        self.collections_dnn_hidden_output = self.model_name + "_collections_dnn_hidden_output"

        self.prada_extra_meta_config = self.config.get_prada_extra_meta_config()
        self.fg = FgParser(self.config.get_fg_config())

        self.metrics = {}
        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

    def build_graph(self, context, features, feature_columns, labels):
        self.features = features[self.model_name]
        self.feature_columns = feature_columns[self.model_name]
        self.labels = labels[self.model_name]

        self.set_global_step()
        self.inference(self.features, self.feature_columns)

        self.loss()
        self.optimizer(context, self.loss_op)
        self.predictions()
        self.mark_output(self.ctr_prediction)
        self.summary()

    def inference(self, features, feature_columns):
        def reset_zero(feature):
            if isinstance(feature, tf.Tensor):
                return tf.zeros_like(feature)
            else:
                return tf.SparseTensor(feature.indices, tf.zeros_like(feature.values), feature.dense_shape)

        if 'week' in features:
            features['week'] = reset_zero(features['week'])
        if 'clock' in features:
            features['clock'] = reset_zero(features['clock'])

        self.embedding_layer(features, feature_columns)
        self.sequence_layer()
        self.mism_net()
        self.pisa_net()
        self.user_net()
        self.mlaf_net()
        self.item_net()
        self.bias_net()
        self.main_net()

    def predictions(self):
        with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
            self.ctr_prediction = tf.identity(self.ctr_prop, name="ctr_prop")

    def loss(self):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            self.reg_loss_f()
            self.multi_sid_gen_loss()

            self.ctr_exp2click_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.labels, logits=self.ctr_logits))

            with tf.name_scope("dual_loss"):
                if self.model_conf['model_hyperparameter'].get('dual_l2_norm', False):
                    self.item_query = layers.layer_norm(self.item_query, begin_norm_axis=-1, begin_params_axis=-1)
                    self.user_query_au_vec = layers.layer_norm(self.user_query_au_vec, begin_norm_axis=-1, begin_params_axis=-1)

                item_query_stop = tf.stop_gradient(tf.identity(self.item_query, name='item_query_stop'))

                self.dual_loss = tf.reduce_mean(tf.pow(tf.multiply(self.labels, self.user_query_au_vec-item_query_stop),2))

                if self.model_conf['model_hyperparameter'].get('nce_loss', False):
                    nce_temperature = self.model_conf['model_hyperparameter'].get('nce_temperature', 0.07)
                    self.dual_loss = self.nce_loss(self.user_query_au_vec, item_query_stop, self.labels, nce_temperature)

            dual_loss_w = self.model_conf['model_hyperparameter'].get("dual_loss_w", 1)
            multi_sid_loss_w = self.model_conf['model_hyperparameter'].get("multi_sid_loss_w", 1)

            self.loss_op = self.ctr_exp2click_loss + self.reg_loss + dual_loss_w * self.dual_loss + multi_sid_loss_w * self.multi_sid_loss
            return self.loss_op

    def embedding_layer(self, features, feature_columns):
        with tf.variable_scope(name_or_scope="Embedding_Layer",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for block_name in set(self.user_column_blocks + self.item_column_blocks + self.item_query_column_blocks + self.bias_column_blocks \
                                  + list(self.seq_column_len.values()) + self.user_profile_column_blocks):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.layer_dict[block_name] = layers.input_from_feature_columns(features,feature_columns[block_name],scope=scope)

            self.sequence_layer_dict = self.build_sequence(self.seq_column_blocks, self.seq_column_len, "seq")

            self.tmsid_3_raw = self.features["emb_tmsid_3"]
            self.tmsid_3_label = self.sparse_to_raw(self.tmsid_3_raw, "0")

    def sequence_layer(self):
        for block_name in self.sequence_layer_dict.keys():
            with arg_scope(model_arg_scope(
                    weight_decay=self.model_conf['model_hyperparameter'].get('attention_l2_reg', 0.0)
            )):
                with tf.variable_scope(name_or_scope="Share_Sequence_Layer_{}".format(block_name),
                                       partitioner=partitioned_variables.min_max_variable_partitioner(
                                           max_partitions=self.config.get_job_config("ps_num"),
                                           min_slice_size=self.config.get_job_config("dnn_min_slice_size")),
                                       reuse=tf.AUTO_REUSE) as scope:

                    max_len = self.fg.get_seq_len_by_sequence_name(block_name)
                    sequence = self.sequence_layer_dict[block_name]

                    sequence_length = self.layer_dict[self.seq_column_len[block_name]]
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)

                    if self.model_conf['model_hyperparameter'].get('self_attention', True):
                        self_poly_vec = attention(queries=sequence,
                                                  keys=sequence,
                                                  values=sequence,
                                                  num_units=None,
                                                  num_output_units=None,
                                                  activation_fn=None,
                                                  normalizer_fn=None,
                                                  normalizer_params=None,
                                                  scope="self_attention",
                                                  reuse=tf.AUTO_REUSE,
                                                  query_masks=sequence_mask,
                                                  key_masks=sequence_mask,
                                                  num_heads=1,
                                                  need_linear_transform=False)
                        self.layer_dict[block_name + "_sa"] = tf.reduce_mean(self_poly_vec, axis=1)
                        self.layer_dict[block_name] = self_poly_vec

    def pisa_net(self):
        with tf.name_scope("PISA_AU_NET"):
            user_profile_layer = []
            for block_name in self.user_profile_column_blocks:
                if block_name not in self.layer_dict:
                    raise ValueError('[user_au_net, layer dict] does not has block : {}'.format(block_name))
                user_profile_layer.append(self.layer_dict[block_name])

            user_au_vec_init = tf.concat(values=user_profile_layer, axis=1)

            with self.variable_scope(name_or_scope="{}_User_Au_Item_Query".format(self.model_name),
                                     partitioner=partitioned_variables.min_max_variable_partitioner(
                                         max_partitions=self.config.get_job_config("ps_num"),
                                         min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                     ):
                self.user_query_au_vec = layers.linear(
                    user_au_vec_init,
                    120,
                    scope="user_query_au_net",
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output],
                    biases_initializer=None)
                if self.model_conf['model_hyperparameter'].get('dual_mlp', False):
                    self.user_query_au_vec = self.mlp_layer(user_au_vec_init, [256,120], True, True, True,"user_query_au_net")
                if self.model_conf['model_hyperparameter'].get('dual_film', True):
                    self.user_query_au_vec = self.film_layer(user_au_vec_init, 120)

    def mism_net(self):
        self.vocab_size = 700
        self.interest_nums = self.model_conf['model_hyperparameter'].get("interest_nums", 30)

        with self.variable_scope(name_or_scope="{}_Mism_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            user_sa_vec = self.layer_dict["user_seq_list_sa"]
            predicted_sid = self.mlp_layer(user_sa_vec, [512, 256], True, True, True, "Semantic_Routing_1")
            self.predicted_sid = self.mlp_layer(predicted_sid, [self.vocab_size], True, False,True, "Semantic_Routing_2")
            self.topk_interests = self.topk_sid_lookup(self.predicted_sid, k=self.interest_nums)

    def user_net(self):
        user_net_layer = []
        for block_name in set(self.user_column_blocks):
            user_net_layer.append(self.layer_dict[block_name])

        with self.variable_scope(name_or_scope="{}_User_Score_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            self._user_net = tf.concat(values=user_net_layer, axis=-1)

            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                for layer_id, num_hidden_units in enumerate(
                        self.model_conf['model_hyperparameter']['dnn_hidden_units']):
                    with self.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        self._user_net = layers.fully_connected(
                            self._user_net,
                            num_hidden_units,
                            getActivationFunctionOp(
                                self.model_conf['model_hyperparameter']['activation']),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True, "is_training": self.is_training})

    def mlaf_net(self):
        with tf.name_scope("MLAF_TA1"):
            u_sa_emb = self.layer_dict["user_seq_list"]
            target_Q_expanded = tf.expand_dims(self.user_query_au_vec, 1)
            att_scores = tf.matmul(target_Q_expanded, u_sa_emb, transpose_b=True)
            att_weights = tf.nn.softmax(att_scores, axis=-1)
            target_attended = tf.squeeze(tf.matmul(att_weights, u_sa_emb), axis=1)

            target_attended = target_attended + self.user_query_au_vec
            target_attended = layers.layer_norm(target_attended, begin_norm_axis=-1, begin_params_axis=-1)

            self.user_ta_vec = tf.identity(target_attended, name='user_ta_vec')
        with tf.name_scope("MLAF_TA2"):
            query = tf.expand_dims(self.user_ta_vec, axis=1)
            key = self.topk_interests
            value = self.topk_interests

            d_model = key.get_shape().as_list()[-1]
            query = layers.linear(query, d_model, scope="query_projection")

            dk = tf.cast(d_model, tf.float32)
            scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(dk)
            aw = tf.nn.softmax(scores, axis=-1)
            mlaf_ta = tf.matmul(aw, value)
            mlaf_ta = tf.squeeze(mlaf_ta, axis=1)

            user_query_au = self.mlp_layer(self.user_query_au_vec, [128], True, True, True, "user_au_mlp")
            mlaf_ta = mlaf_ta + user_query_au

        with tf.name_scope("MLAF_FINAL_UNION"):
            union_type = self.model_conf['model_hyperparameter'].get("union_type", "mlaf_double_gate")
            gate_type = self.model_conf['model_hyperparameter'].get("gate_type", "element_level")

            user_vec_dim = self._user_net.get_shape().as_list()[-1]

            if union_type == "mlaf_single_gate":
                with tf.name_scope("mlaf_single_gate"):
                    gate_input = tf.concat([mlaf_ta, self._user_net], axis=1)
                    gate_output_dim = user_vec_dim if gate_type == "element_level" else 1
                    w = layers.fully_connected(
                        gate_input,
                        gate_output_dim,
                        activation_fn=tf.nn.sigmoid,
                        scope="gate_weight")
                    user_vec = w * mlaf_ta + (1.0 - w) * self._user_net
            elif union_type == "mlaf_double_gate":
                with tf.name_scope("mlaf_double_gate"):
                    gate_input = tf.concat([mlaf_ta, self._user_net], axis=1)
                    gate_output_dim = user_vec_dim if gate_type == "element_level" else 1
                    w_mlaf = layers.fully_connected(
                        gate_input,
                        gate_output_dim,
                        activation_fn=tf.nn.sigmoid,
                        scope="gate_mlaf")
                    w_user = layers.fully_connected(
                        gate_input,
                        gate_output_dim,
                        activation_fn=tf.nn.sigmoid,
                        scope="gate_user")
                    w_sum = w_mlaf + w_user + 1e-8
                    w_mlaf_norm = w_mlaf / w_sum
                    w_user_norm = w_user / w_sum
                    user_vec = w_mlaf_norm * mlaf_ta + w_user_norm * self._user_net
            elif union_type == "mlaf_bilinear":
                with tf.name_scope("mlaf_bilinear"):
                    bilinear = tf.einsum('bi,bj->bij', mlaf_ta, self._user_net)
                    bilinear = tf.reshape(bilinear, [-1, user_vec_dim * user_vec_dim])
                    user_vec = layers.fully_connected(
                        bilinear,
                        user_vec_dim,
                        activation_fn=tf.nn.relu,
                        scope="bilinear_compress")

            self.user_vec = tf.identity(user_vec, name='user_vec')

    def item_net(self):
        item_query_net_layer = []
        for block_name in self.item_query_column_blocks:
            if block_name not in self.layer_dict:
                raise ValueError('[item query net, layer dict] does not has block : {}'.format(block_name))
            item_query_net_layer.append(self.layer_dict[block_name])
        self.item_query = tf.concat(values=item_query_net_layer, axis=1)

        item_net_layer = []
        for block_name in self.item_column_blocks:
            if block_name not in self.layer_dict:
                raise ValueError('[item net, layer dict] does not has block : {}'.format(block_name))
            item_net_layer.append(self.layer_dict[block_name])
        with self.variable_scope(name_or_scope="{}_Item_Score_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            self._item_net = tf.concat(values=item_net_layer, axis=1)

            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                for layer_id, num_hidden_units in enumerate(
                        self.model_conf['model_hyperparameter']['dnn_hidden_units']):
                    with self.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as item_hidden_layer_scope:
                        self._item_net = layers.fully_connected(
                            self._item_net,
                            num_hidden_units,
                            getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                            scope=item_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True, "is_training": self.is_training})

            self.item_vec = tf.identity(self._item_net, name='item_vec')

    def bias_net(self):
        if len(self.bias_column_blocks) <= 0:
            return
        bias_net_layer = []
        for block_name in self.bias_column_blocks:
            if block_name not in self.layer_dict:
                raise ValueError('[Bias net, layer dict] does not has block : {}'.format(block_name))
            bias_net_layer.append(self.layer_dict[block_name])
        with self.variable_scope(name_or_scope="{}_Bias_Score_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            bias_net = tf.concat(values=bias_net_layer, axis=1)
            with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                for layer_id, num_hidden_units in enumerate(
                        self.model_conf['model_hyperparameter']['bias_dnn_hidden_units']):
                    with self.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        bias_net = layers.fully_connected(
                            bias_net,
                            num_hidden_units,
                            getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.collections_dnn_hidden_layer],
                            outputs_collections=[self.collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm,
                            normalizer_params={"scale": True, "is_training": self.is_training})

                if self.model_conf['model_hyperparameter']['need_dropout']:
                    bias_net = tf.layers.dropout(
                        bias_net,
                        rate=self.model_conf['model_hyperparameter']['dropout_rate'],
                        noise_shape=None,
                        seed=None,
                        training=self.is_training,
                        name=None)

                bias_logits = layers.linear(
                    bias_net,
                    1,
                    scope="bias_net",
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output],
                    biases_initializer=None)
                bias_tensor = tf.identity(bias_logits, name='bias')
                shape = bias_tensor.get_shape().as_list()
                bias_placeholder = tf.placeholder(tf.float32, shape=shape, name="bias_placeholder")

            self.ctr_bias_logits = bias_tensor

    def main_net(self):
        with (self.variable_scope(name_or_scope="{}_Main_Score_Network".format(self.model_name),
                                  partitioner=partitioned_variables.min_max_variable_partitioner(
                                      max_partitions=self.config.get_job_config("ps_num"),
                                      min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                  )):

            user_vec = tf.nn.l2_normalize(self.user_vec, dim = 1)
            item_vec = tf.nn.l2_normalize(self.item_vec, dim = 1)
            ctr_logits = tf.reduce_sum(tf.multiply(user_vec, item_vec), -1)
            ctr_logits = tf.reshape(ctr_logits, [-1, 1])

            if len(self.bias_column_blocks) > 0:
                ctr_logits += self.ctr_bias_logits

            bias = contrib_variables.model_variable(
                "bias_weight",
                shape=[1],
                initializer=tf.zeros_initializer(),
                trainable=True)

            self.ctr_logits = nn_ops.bias_add(ctr_logits, bias)
            self.ctr_prop = tf.sigmoid(self.ctr_logits)

    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.reduce_sum(reg_losses)

    def summary(self):
        metrics_dict = {}
        with tf.name_scope("{}_Metrics".format(self.model_name)):
            worker_device = "/job:worker/task:{}".format(self.context.get_task_id())
            with tf.device(worker_device):
                decay_rate = self.algo_config.get('auc', {}).get('decay_rate', 0.999)
                self.current_ctr_auc, self.total_ctr_auc = auc(labels=self.labels,
                                                               predictions=self.ctr_prediction,
                                                               num_thresholds=2000,
                                                               name=self.model_name + '-auc')
                current_ctr_auc_decay, update_ctr_auc_decay = metrics.auc(
                    labels=self.labels,
                    predictions=self.ctr_prediction,
                    num_thresholds=2000,
                    name=self.model_name + '-decay_ctr_auc-' + str(decay_rate),
                    decay_rate=decay_rate)
                with tf.control_dependencies([update_ctr_auc_decay]):
                    current_ctr_auc_decay = tf.identity(current_ctr_auc_decay)
                metrics_dict.update({
                    'scalar/current_ctr_auc': self.current_ctr_auc,
                    'scalar/total_ctr_auc': self.total_ctr_auc,
                    'scalar/current_ctr_auc_decay-' + str(decay_rate): current_ctr_auc_decay,
                    'scalar/update_ctr_auc_decay-' + str(decay_rate): update_ctr_auc_decay,
                })

        metrics_dict['scalar/loss'] = self.loss_op
        metrics_dict['scalar/reg_loss'] = self.reg_loss
        metrics_dict['scalar/ctr_exp2click_loss'] = self.ctr_exp2click_loss
        metrics_dict['scalar/multi_sid_loss'] = self.multi_sid_loss
        metrics_dict['scalar/dual_loss'] = self.dual_loss
        metrics_dict['scalar/label_mean'] = tf.reduce_mean(self.labels)
        metrics_dict['scalar/ctr_prediction_mean'] = tf.reduce_mean(self.ctr_prediction)

        recall_50, precision_50 = self.compute_topk_recall_and_precision(self.ctr_logits, self.labels, 50)
        recall_100, precision_100 = self.compute_topk_recall_and_precision(self.ctr_logits, self.labels, 100)
        metrics_dict['scalar/recall@50'] = recall_50
        metrics_dict['scalar/recall@100'] = recall_100
        metrics_dict['scalar/precision@50'] = precision_50
        metrics_dict['scalar/precision@100'] = precision_100

        self.metrics['scalar/sid_hitrate@1'] = self.sid_hitrate_1
        self.metrics['scalar/sid_hitrate@5'] = self.sid_hitrate_5
        self.metrics['scalar/sid_hitrate@10'] = self.sid_hitrate_10
        self.metrics['scalar/sid_hitrate@15'] = self.sid_hitrate_15
        self.metrics['scalar/sid_hitrate@20'] = self.sid_hitrate_20
        self.metrics['scalar/sid_hitrate@25'] = self.sid_hitrate_25
        self.metrics['scalar/sid_hitrate@30'] = self.sid_hitrate_30
        self.metrics['scalar/sid_hitrate@50'] = self.sid_hitrate_50

        self.metrics.update(metrics_dict)
        with tf.name_scope("{}_Metrics_Scalar".format(self.model_name)):
            for key, metric in self.metrics.items():
                tf.summary.scalar(name=key, tensor=metric)

        with tf.name_scope("{}_Layer_Summary".format(self.model_name)):
            add_norm2_summary(self.collections_dnn_hidden_layer)
            add_dense_output_summary(self.collections_dnn_hidden_output)
            add_weight_summary(self.collections_dnn_hidden_layer)

        self.sample_trace_dict = {}
        self.add_sample_trace_dict("prediction", self.ctr_prediction)
        self.add_sample_trace_dict("id", self.features['id'])
        self.add_sample_trace_dict("label", self.features['label'])
        self.context.get_model_ops().set_sample_trace_dict(self.sample_trace_dict)
        self.context.get_model_ops().set_global_step(self.global_step)
        return self.metrics

    def add_sample_trace_dict(self, key, value):
        try:
            self.sample_trace_dict[key] = tf.sparse_tensor_to_dense(value, default_value="")
        except:
            self.sample_trace_dict[key] = value

    def optimizer(self, loss_op):
        with tf.variable_scope(
                name_or_scope="Optimize",
                partitioner=partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.config.get_job_config("ps_num"),
                    min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                ),
                reuse=tf.AUTO_REUSE):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = self.update_op(name=self.model_name)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                if 'scope' not in opt_conf or opt_conf["scope"] == "Global":
                    global_opt_name = opt_name
                    global_optimizer = optimizer
                else:
                    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opt_conf["scope"])
                    if len(vars) != 0:
                        for var in vars:
                            if var in global_opt_vars:
                                global_opt_vars.remove(var)
                        train_op, _, _ = myopt.optimize_loss(
                            loss=loss_op,
                            global_step=self.global_step,
                            learning_rate=opt_conf.get("learning_rate", 0.01),
                            optimizer=optimizer,
                            clip_gradients=opt_conf.get('clip_gradients', 5),
                            variables=vars,
                            increment_global_step=False,
                            summaries=myopt.OPTIMIZER_SUMMARIES)
                        train_ops.append(train_op)
            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op

    def update_op(self, name):
        update_ops = []
        start = ('Share') if name is None else ('Share', name)
        for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            if update_op.name.startswith(start):
                update_ops.append(update_op)
        return update_ops

    def get_optimizer(self, opt_name, opt_conf, global_step):
        optimizer = None
        for name in optimizer_dict:
            if opt_name == name:
                optimizer = optimizer_dict[name](opt_conf, global_step)
                break
        return optimizer

    def set_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def mark_output(self, ctr_prediction):
        with tf.name_scope("CTR_Mark_Output"):
            prediction = tf.identity(ctr_prediction, name="rank_predict")

    def run_train(self, mon_session, task_index):
        localcnt = 0
        NanInfNum = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step, self.loss_op, self.metrics, self.labels, self.localvar]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    global_step, loss, metrics, labels, flocalv = mon_session.run(
                        run_ops, feed_dict=feed_dict)
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.train_ops)
                    global_step, loss, metrics, labels, flocalv, _ = mon_session.run(
                        run_ops, feed_dict=feed_dict)

                    if len(self.localvar) > 0:
                        index = np.array([0, -1])

                    ctr_click_auc, total_ctr_click_auc = metrics['scalar/current_ctr_auc'], metrics['scalar/total_ctr_auc']

                    newmark = np.max(flocalv[0][np.array([0, -1])])
                    if newmark > 20000:
                        index = np.array([0, -1])
                        flocalv = mon_session.run(self.reset_auc_ops, feed_dict=feed_dict)

            except (ResourceExhaustedError, OutOfRangeError) as e:
                break
            except SkipNanInfException as e:
                NanInfNum += 1
                continue
            except ConnectionError as e:
                pass
            except Exception as e:
                pass

    def build_sequence(self, seq_column_blocks, seq_column_len, name):
        features = self.features
        feature_columns = self.feature_columns
        sequence_layer_dict = {}
        if seq_column_blocks is None or len(seq_column_blocks) == 0:
            return
        with tf.variable_scope(name_or_scope='{}_seq_input_from_feature_columns'.format(name),
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config('ps_num'),
                                   min_slice_size=self.config.get_job_config('embedding_min_slice_size')),
                               reuse=tf.AUTO_REUSE) as (scope):
            if len(seq_column_blocks) > 0:
                for block_name in seq_column_blocks:
                    if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                        raise ValueError('block_name:(%s) not in feature_columns for seq' % block_name)
                    seq_len = self.fg.get_seq_len_by_sequence_name(block_name)

                    sequence_stack = _input_from_feature_columns(features,
                                                                 feature_columns[block_name],
                                                                 weight_collections=None,
                                                                 trainable=True,
                                                                 scope=scope,
                                                                 output_rank=3,
                                                                 default_name='sequence_input_from_feature_columns')
                    sequence_stack = tf.reshape(sequence_stack, [-1, seq_len, sequence_stack.get_shape()[(-1)].value])
                    sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                    if block_name in seq_column_len and seq_column_len[block_name] in self.layer_dict:
                        sequence_length = self.layer_dict[seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                        sequence_stack = tf.reshape(
                            tf.where(tf.reshape(sequence_mask, [-1]), sequence_2d, tf.zeros_like(sequence_2d)),
                            tf.shape(sequence_stack))
                    else:
                        sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                    sequence_layer_dict[block_name] = sequence_stack
        return sequence_layer_dict

    def topk_sid_lookup(self, predicted_sid, k=25):
        with tf.name_scope("TopKSidLookup"):
            topk_scores, topk_indices = tf.nn.top_k(
                predicted_sid,
                k=k,
                sorted=True,
                name="TopK_Scores_Indices"
            )

            topk_indices = topk_indices + 1

            dense_sid_id = tf.reshape(tf.as_string(topk_indices), [-1, 1])

            indices = tf.where(tf.not_equal(dense_sid_id, "-1"))
            values = tf.gather_nd(dense_sid_id, indices)
            dense_shape = tf.shape(dense_sid_id, out_type=tf.int64)
            sparse_sid_id = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

            with self.variable_scope("Interest_Embedding_Layer") as scope:
                self.top_sid_emb = layers.input_from_feature_columns(columns_to_tensors={"emb_tmsid_3_": sparse_sid_id},
                                                                     feature_columns=self.feature_columns['tmsid_columns'],
                                                                     scope=scope)

            topk_interests = tf.reshape(self.top_sid_emb, [-1, self.interest_nums, self.top_sid_emb.get_shape().as_list()[-1]])

            self.topk_scores = topk_scores
            self.topk_indices = topk_indices

            return topk_interests

    def mlp_layer(self, dnn_input, dnn_hidden_units, batch_norm=True, need_activation = True, need_dropout=True, name_scope="default"):
        normalizer_fn = None
        normalizer_params = None
        if batch_norm:
            normalizer_fn = layers.batch_norm
            normalizer_params = {
                "scale": True,
                "is_training": self.is_training
            }
        with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with self.variable_scope(
                        "{}_dnn_hiddenlayer_{}".format(name_scope, layer_id)) as dnn_hidden_layer_scope:
                    dnn_input = layers.fully_connected(
                        dnn_input,
                        num_hidden_units,
                        getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']) if need_activation else None,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        scope=dnn_hidden_layer_scope,
                        variables_collections=[self.collections_dnn_hidden_layer],
                        outputs_collections=[self.collections_dnn_hidden_output]
                    )
            if need_dropout:
                dnn_input = tf.layers.dropout(dnn_input, rate=self.model_conf['model_hyperparameter']['dropout_rate'], training=self.is_training)
        return dnn_input

    def film_layer(self, user_attrs, dim):
        with tf.variable_scope(name_or_scope="hyper_aug",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            seed_vector = tf.get_variable("global_seed", [1, dim],
                                          initializer=tf.glorot_uniform_initializer())

            h = tf.layers.dense(user_attrs, 128, activation=tf.nn.relu)

            params = tf.layers.dense(h, 2 * dim, activation=None)

            gamma, beta = tf.split(params, num_or_size_splits=2, axis=1)

            seed_broadcast = tf.tile(seed_vector, [tf.shape(user_attrs)[0], 1])

            seed_broadcast = self.mlp_layer(seed_broadcast, [256, dim], True, True, True, name_scope="seed_mlp")

            a_u = gamma * seed_broadcast + beta

            return a_u

    def nce_loss(self, vec1, vec2, label, temperature):
        similarity_matrix = tf.matmul(vec1, vec2, transpose_b=True)
        similarity_matrix = similarity_matrix / temperature

        batch_size = tf.shape(similarity_matrix)[0]
        positive_labels = tf.eye(batch_size)

        nce_loss_per_sample = tf.nn.softmax_cross_entropy_with_logits(
            labels=positive_labels,
            logits=similarity_matrix
        )

        valid_mask = tf.squeeze(tf.cast(label, tf.float32), axis=-1)
        nce_loss_per_sample = nce_loss_per_sample * valid_mask

        num_positive = tf.reduce_sum(valid_mask)
        num_positive = tf.maximum(num_positive, 1.0)

        nce_loss = tf.reduce_sum(nce_loss_per_sample) / num_positive

        return nce_loss

    def multi_sid_gen_loss(self):
        with tf.name_scope("multi_sid_gen_loss"):
            self.label_sid = tf.strings.to_number(self.tmsid_3_label, out_type=tf.int64)
            if len(self.label_sid.shape) == 2:
                self.label_sid = tf.squeeze(self.label_sid, axis=-1)

            self.label_sid = self.label_sid - 1

            self.multi_sid_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label_sid,
                    logits=self.predicted_sid
                )
            )

            def calculate_hr_at_k(targets, predictions, k):
                in_top_k = tf.nn.in_top_k(predictions=predictions, targets=targets, k=k)
                return tf.reduce_mean(tf.cast(in_top_k, tf.float32))

            self.sid_hitrate_1 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 1)
            self.sid_hitrate_5 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 5)
            self.sid_hitrate_10 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 10)
            self.sid_hitrate_15 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 15)
            self.sid_hitrate_20 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 20)
            self.sid_hitrate_25 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 25)
            self.sid_hitrate_30 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 30)
            self.sid_hitrate_50 = calculate_hr_at_k(self.label_sid, self.predicted_sid, 50)

    def compute_topk_recall_and_precision(self, logits, labels, topk):
        logits_1d = tf.reshape(logits, [-1])
        labels_1d = tf.reshape(labels, [-1])
        _, indices_of_ranks = tf.nn.top_k(logits_1d, k=topk)
        topk_labels = tf.gather(labels_1d, indices_of_ranks)
        recall = tf.reduce_sum(topk_labels) / tf.reduce_sum(labels_1d)
        precision = tf.reduce_mean(topk_labels)
        return recall, precision
        def variable_scope(self, *args, **kwargs):
            kwargs['partitioner'] = partitioned_variables.min_max_variable_partitioner(
                max_partitions=self.config.get_job_config("ps_num"),
                min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        kwargs['reuse'] = tf.AUTO_REUSE
        return tf.variable_scope(*args, **kwargs)

    def get_fc_params(self):
        return {
            "normalizer_fn": layers.batch_norm if self.algo_config.get('batch_norm', True) else None,
            "normalizer_params": {"scale": True,
                                  "is_training": self.is_training,
                                  "epsilon": self.algo_config.get('batch_norm_epsilon', 0.001),
                                  "decay": self.algo_config.get('batch_norm_decay', 0.999)},
            "activation_fn": getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation']),
            "variables_collections": [self.collections_dnn_hidden_layer],
            "outputs_collections": [self.collections_dnn_hidden_output],
        }

    def sparse_to_raw(self, sparse_tensor, default_sc):
        tensor = tf.sparse_tensor_to_dense(sparse_tensor, default_value=default_sc)
        tensor = tf.reduce_join(tensor, axis=1)
        tensor = tf.reshape(tensor, [-1, 1])
        tensor = tf.cast(tensor, tf.string)
        return tensor