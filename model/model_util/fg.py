from collections import OrderedDict
import json


FEATURE_NAME_KEY = "feature_name"
SEQUENCE_NAME_KEY = "sequence_name"
FEATURE_KEY = "features"
SEQUENCE_LENGTH_KEY = "sequence_length"
value_type_key = 'value_type'
feature_type_key = 'feature_type'
feature_name_key = 'feature_name'


class FgParser(object):
  def __init__(self, fg_config):
    self.fg_config = fg_config
    self.feature_conf_map = {}
    self.seq_feature_conf_map = {}
    self.seq_len_dict = {}
    self._parse_feature_conf(self.fg_config)

  def _parse_feature_conf(self, config):
    self._feature_conf_map = OrderedDict()
    feature_conf_list = config[FEATURE_KEY]
    for feature_conf in feature_conf_list:
      if "_comment" in feature_conf:
        continue
      feature_name = feature_conf[feature_name_key]
      if feature_conf.has_key('sequence_name'):
        self.seq_feature_conf_map[feature_name] = feature_conf
        self.seq_len_dict[feature_conf['sequence_name']] = feature_conf[SEQUENCE_LENGTH_KEY]
      else:
        self.feature_conf_map[feature_name] = feature_conf

  def get_seq_len_by_sequence_name(self, name):
    return self.seq_len_dict[name]

  def get_bucket_size_by_fc_name(self, name):
    return self.feature_conf_map[name]['hash_bucket_size']

  def get_emb_dim_by_fc_name(self, name):
    return self.feature_conf_map[name]['embedding_dimension']

  def get_shared_name_by_fc_name(self, name):
    return self.feature_conf_map[name]['shared_name']

  def get_feature_conf_by_fc_name(self, name):
    return self.feature_conf_map[name]

  def extract_src_embedding_config(self, import_embedding_config, embedding_service_meta):
    '''
    example of arguments:
  import_embedding_config
        {
      "embedding_name": "week_shared_embedding",
      "import_type": "ckpt",
      "embedding_table": "embedding_table_1",
      "trainable": false
  }

      {
            "embedding_table_1": {
              "embedding_variable_meta_map": {
          "week_shared_embedding": {
          "embedding_var_keys": [
              "input_from_feature_columns/week_shared_embedding/week/part_0-keys"
          ],
          "weights_op_path": "input_from_feature_columns/week_shared_embedding/week",
          "embedding_var_values": [
              "input_from_feature_columns/week_shared_embedding/week/part_0-values"
          ],
          "publish_mode": {
              "ckpt": {
              "path": "/home/wuul.lwy/dev/local_debug/ckpt_dir_1"
              }
          }
          }
              }
      }
        }

    expected output
    {
        "type": "ckpt", // candidates is ['ckpt', 'igraph']
        "partition_num": 1, // for ckpt
        "src_scope": "input_from_feature_columns/week_shared_embedding/week", // for ckpt
        "ckpt_path": "/home/wuul.lwy/dev/local_debug/ckpt_dir_1" // for ckpt
        "igraph_table_name": "todo", // for igraph
        "trainable": false  // for ckpt and igraph
    }
    '''
    if 'embedding_name' not in import_embedding_config or 'embedding_table' not in import_embedding_config:
      raise Exception('invalid import_embedding_config:\n%s' % (json.dumps(import_embedding_config, indent=2)))
    embedding_name = import_embedding_config['embedding_name']
    embedding_table = import_embedding_config['embedding_table']
    root = embedding_service_meta
    if embedding_table not in root:
      raise Exception('no corresponding embedding table %s in [%s]' % (embedding_table, str(root.keys())))
    root = root[embedding_table]['embedding_variable_meta_map']
    if embedding_name not in root:
      raise Exception('no corresponding embedding %s in [%s]' % (embedding_name, str(root.keys())))
    ori_embedding_config = root[embedding_name]
    import_type = import_embedding_config.get('import_type', 'ckpt')
    if import_type == 'ckpt':
      src_scope = ori_embedding_config['weights_op_path']
      src_scope = src_scope if src_scope.endswith('/') else src_scope + '/'
      src_embedding_config = {
        "type": "ckpt",
        "partition_num": len(ori_embedding_config['embedding_var_keys']),
        "src_scope": src_scope,
        "ckpt_path": ori_embedding_config['publish_mode']['ckpt']['path'],
        "trainable": import_embedding_config.get('trainable', True),
        "always_load_from_specific_ckpt": import_embedding_config.get('always_load_from_specific_ckpt', False)
      }
    elif import_type == 'igraph':
      src_embedding_config = {
        "type": "igraph",
        "igraph_table_name": ori_embedding_config['publish_mode']['igraph']['table_name'],
        "trainable": import_embedding_config.get('trainable', True),
        "always_load_from_specific_ckpt": import_embedding_config.get('always_load_from_specific_ckpt', False)
      }
    else:
      raise Exception("unsupported embedding import type %s" % (import_type))
    return src_embedding_config