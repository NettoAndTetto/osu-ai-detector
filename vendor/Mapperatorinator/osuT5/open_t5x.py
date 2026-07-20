import torch
from flax import serialization
import tensorstore as ts
import numpy as np
import jax
from transformers import T5Config, T5ForConditionalGeneration


def _maybe_update_ts_from_gcs_to_file(ckpt_contents):
    """Updates the TensorStore driver to gfile or file if different."""

    # if saved in gcs, change to file
    def _gcs_to_file_driver(arr_or_ts_spec_dict):
        if not isinstance(arr_or_ts_spec_dict, dict):
            return arr_or_ts_spec_dict

        if arr_or_ts_spec_dict['kvstore']['driver'] == 'gcs':
            ts_spec_dict = arr_or_ts_spec_dict
            path = ts_spec_dict['kvstore'].pop('path')
            driver = 'file'
            ts_spec_dict['kvstore'] = {'path': path, 'driver': driver}
        elif arr_or_ts_spec_dict['kvstore']['driver'] == 'gfile':
            ts_spec_dict = arr_or_ts_spec_dict
            driver = 'file'
            ts_spec_dict['kvstore']['driver'] = driver
            ts_spec_dict['kvstore']['path'] = ckpt_dir + ts_spec_dict['kvstore']['path']
        elif arr_or_ts_spec_dict['kvstore']['driver'] == 'file':
            ts_spec_dict = arr_or_ts_spec_dict
        else:
            raise ValueError(
                'Unsupported TensoreStore driver. Got '
                f'{arr_or_ts_spec_dict["kvstore"]["driver"]}.'
            )

        return ts_spec_dict

    def _is_leaf(value):
        return not isinstance(value, dict) or set(value.keys()) >= {
            'driver',
            'kvstore',
            'metadata',
        }

    return jax.tree_util.tree_map(
        _gcs_to_file_driver, ckpt_contents, is_leaf=_is_leaf
    )


ckpt_dir = "mt3\\checkpoints\\mt3\\"
ckpt_path = ckpt_dir + "checkpoint"

with open(ckpt_path, 'rb') as fp:
    ckpt_contents = serialization.msgpack_restore(fp.read())

ckpt_contents = _maybe_update_ts_from_gcs_to_file(ckpt_contents)


def _is_leaf2(value):
    return not isinstance(value, dict) or set(value.keys()) >= {
        'driver',
        'kvstore',
        'metadata',
    }


def open_ts(leaf):
    if not isinstance(leaf, dict):
        return leaf
    return np.array(ts.open(leaf, open=True).result())


test = jax.tree_util.tree_map(
    open_ts, ckpt_contents, is_leaf=_is_leaf2
)

config = T5Config.from_pretrained("google/t5-v1_1-small")
model = T5ForConditionalGeneration(config)


state_dict = {}
for stack in ["encoder", "decoder"]:
    for layer in range(8):
        self_attention_name = 'self_attention' if stack == 'decoder' else 'attention'
        attention_norm_name = 'pre_self_attention_layer_norm' if stack == 'decoder' else 'pre_attention_layer_norm'
        i = 0
        state_dict[f"{stack}.block.{layer}.layer.{i}.SelfAttention.q.weight"] = test['optimizer']['target'][stack][f'layers_{layer}'][self_attention_name]['query']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.SelfAttention.k.weight"] = test['optimizer']['target'][stack][f'layers_{layer}'][self_attention_name]['key']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.SelfAttention.v.weight"] = test['optimizer']['target'][stack][f'layers_{layer}'][self_attention_name]['value']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.SelfAttention.o.weight"] = test['optimizer']['target'][stack][f'layers_{layer}'][self_attention_name]['out']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.layer_norm.weight"] = test['optimizer']['target'][stack][f'layers_{layer}'][attention_norm_name]['scale']
        if stack == "decoder":
            i += 1
            state_dict[f"{stack}.block.{layer}.layer.{i}.EncDecAttention.q.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['encoder_decoder_attention']['query']['kernel']
            state_dict[f"{stack}.block.{layer}.layer.{i}.EncDecAttention.k.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['encoder_decoder_attention']['key']['kernel']
            state_dict[f"{stack}.block.{layer}.layer.{i}.EncDecAttention.v.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['encoder_decoder_attention']['value']['kernel']
            state_dict[f"{stack}.block.{layer}.layer.{i}.EncDecAttention.o.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['encoder_decoder_attention']['out']['kernel']
            state_dict[f"{stack}.block.{layer}.layer.{i}.layer_norm.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['pre_cross_attention_layer_norm']['scale']
        i += 1
        state_dict[f"{stack}.block.{layer}.layer.{i}.DenseReluDense.wi_0.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['mlp']['wi_0']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.DenseReluDense.wi_1.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['mlp']['wi_1']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.DenseReluDense.wo.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['mlp']['wo']['kernel']
        state_dict[f"{stack}.block.{layer}.layer.{i}.layer_norm.weight"] = test['optimizer']['target'][stack][f'layers_{layer}']['pre_mlp_layer_norm']['scale']
    state_dict[f"{stack}.final_layer_norm.weight"] = test['optimizer']['target'][stack][f'{stack}_norm']['scale']

# Convert to tensors
for k, v in state_dict.items():
    x = torch.tensor(v)
    if x.dim() == 2:
        x = x.T
    state_dict[k] = x

model.load_state_dict(state_dict, strict=False)
torch.save(model.state_dict(), ckpt_dir + "pytorch_model.bin")

print("done")
