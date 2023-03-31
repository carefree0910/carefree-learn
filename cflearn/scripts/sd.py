import json
import torch

from torch import Tensor
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import shallow_copy_dict
from cftool.array import tensor_dict_type

from ..misc.toolkit import get_tensors
from ..misc.toolkit import download_static
from ..api.cv.diffusion import DiffusionAPI


key_mapping_inv = {}


def _convert_cond_stage(d: tensor_dict_type, md: tensor_dict_type) -> tensor_dict_type:
    def _get_dv(k: str) -> Tensor:
        v = d.get(k)
        if v is None:
            v = d[k.replace(".text_model", "")]
        return v

    if not d or not md:
        return {}
    d, md = map(shallow_copy_dict, [d, md])
    nd = {
        "condition_model.m.token_embedding.weight": _get_dv(
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"
        ),
        "condition_model.m.text_transformer.encoder.pos_encoding.pos_encoding": _get_dv(
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"
        )[None, ...],
        "condition_model.m.text_transformer.encoder.head.norms.0.weight": _get_dv(
            "cond_stage_model.transformer.text_model.final_layer_norm.weight"
        ),
        "condition_model.m.text_transformer.encoder.head.norms.0.bias": _get_dv(
            "cond_stage_model.transformer.text_model.final_layer_norm.bias"
        ),
    }
    for key in [
        "condition_model.m.logit_scale",
        "condition_model.m.text_transformer.attention_mask",
        "condition_model.m.text_projection.weight",
        "condition_model.m.text_projection.bias",
    ]:
        nd[key] = md.pop(key)
    prefix = "cond_stage_model.transformer.text_model.encoder.layers"
    m_prefix = "condition_model.m.text_transformer.encoder.mixing_blocks"
    layer_indices = []
    for key in d:
        if not key.startswith(prefix):
            continue
        try:
            layer_indices.append(int(key.split(".")[5]))
        except:
            continue
    num_cond_layers = max(layer_indices) + 1
    for i in range(num_cond_layers):
        i_prefix = f"{prefix}.{i}"
        m_i_prefix = f"{m_prefix}.{i}"
        get_w = lambda n: _get_dv(f"{i_prefix}.self_attn.{n}_proj.weight")
        get_b = lambda n: _get_dv(f"{i_prefix}.self_attn.{n}_proj.bias")
        in_w = torch.cat(list(map(get_w, ["q", "k", "v"])))
        qkv_bias = torch.cat(list(map(get_b, ["q", "k", "v"])))
        m_i_token_mixing_prefix = f"{m_i_prefix}.token_mixing.net"
        nd[f"{m_i_token_mixing_prefix}.in_w"] = in_w
        nd[f"{m_i_token_mixing_prefix}.qkv_bias"] = qkv_bias
        out_w = _get_dv(f"{i_prefix}.self_attn.out_proj.weight")
        out_b = _get_dv(f"{i_prefix}.self_attn.out_proj.bias")
        nd[f"{m_i_token_mixing_prefix}.out_linear.linear.weight"] = out_w
        nd[f"{m_i_token_mixing_prefix}.out_linear.linear.bias"] = out_b
        get_fc_w = lambda n: _get_dv(f"{i_prefix}.mlp.{n}.weight")
        get_fc_b = lambda n: _get_dv(f"{i_prefix}.mlp.{n}.bias")
        m_i_channel_mixing_prefix = f"{m_i_prefix}.channel_mixing.net"
        nd[f"{m_i_channel_mixing_prefix}.0.linear.weight"] = get_fc_w("fc1")
        nd[f"{m_i_channel_mixing_prefix}.0.linear.bias"] = get_fc_b("fc1")
        nd[f"{m_i_channel_mixing_prefix}.3.linear.weight"] = get_fc_w("fc2")
        nd[f"{m_i_channel_mixing_prefix}.3.linear.bias"] = get_fc_b("fc2")
        for ln, m_ln in zip(
            ["layer_norm1", "layer_norm2"],
            ["token_norm", "channel_norm"],
        ):
            nd[f"{m_i_prefix}.{m_ln}.weight"] = _get_dv(f"{i_prefix}.{ln}.weight")
            nd[f"{m_i_prefix}.{m_ln}.bias"] = _get_dv(f"{i_prefix}.{ln}.bias")
    return nd


def _convert_first_stage(d: tensor_dict_type, md: tensor_dict_type) -> tensor_dict_type:
    d, md = map(shallow_copy_dict, [d, md])
    keys = list(d.keys())
    m_keys = list(md.keys())
    for k in [
        "first_stage.codebook.embedding.weight",
        "first_stage.to_embedding.weight",
        "first_stage.to_embedding.bias",
        "first_stage.from_embedding.weight",
        "first_stage.from_embedding.bias",
    ]:
        if k in m_keys:
            m_keys.remove(k)
            m_keys.append(k)
    start_idx = None
    end_idx = None
    for i, k in enumerate(keys):
        if start_idx is None and k.startswith("first_stage_model.decoder.up"):
            start_idx = i
        elif k == "first_stage_model.decoder.norm_out.weight":
            end_idx = i
    if end_idx is None:
        raise ValueError("`first_stage_model.decoder.norm_out.weight` is missing")
    before = keys[:start_idx]
    up = keys[start_idx:end_idx]
    after = keys[end_idx:]
    num_up = int(keys[end_idx - 1].split(".")[3]) + 1
    new_up = []
    mapping = shallow_copy_dict(d)
    local_key_mapping = {k: k for k in d}
    for k in up:
        mapping.pop(k)
    for i in reversed(range(num_up)):
        for k in up:
            if k.startswith(f"first_stage_model.decoder.up.{i}"):
                ks = k.split(".")
                ks[3] = str(num_up - int(ks[3]))
                new_k = ".".join(ks)
                new_up.append(new_k)
                mapping[new_k] = d[k]
                local_key_mapping[new_k] = k
    keys = before + new_up + after

    def _handle_attn(prefix: str) -> None:
        attn_keys: Dict[int, Dict[int, List[str]]] = {}
        all_attn_keys = []
        for key in keys:
            if key.startswith(prefix):
                ks = key.split(".")
                if ks[4] == "attn":
                    level = int(ks[3])
                    idx = int(ks[5])
                    attn_keys.setdefault(level, {}).setdefault(idx, []).append(key)
                    all_attn_keys.append(key)
        for level, l_attn_keys in attn_keys.items():
            for idx, idx_attn_keys in l_attn_keys.items():
                for i, key in enumerate(keys):
                    if key.startswith(f"{prefix}.{level}.block.{idx + 1}"):
                        for attn_key in idx_attn_keys:
                            keys.remove(attn_key)
                            keys.insert(i, attn_key)
                            i += 1
                        break

    _handle_attn("first_stage_model.encoder.down")
    _handle_attn("first_stage_model.decoder.up")
    new_d = {}
    for k, mk in zip(keys, m_keys):
        v = mapping[k]
        mv = md[mk]
        assert v.shape == mv.shape, f"{k} ({v.shape}) != {mk} ({mv.shape})"
        new_d[mk] = v
        key_mapping_inv[mk] = local_key_mapping[k]
    return new_d


def _convert_others(d: tensor_dict_type, md: tensor_dict_type) -> tensor_dict_type:
    nd = {}
    d, md = map(shallow_copy_dict, [d, md])
    keys = list(d.keys())
    m_keys = list(md.keys())
    for k in reversed(keys):
        if k == "model_ema.decay":
            keys.remove(k)
        elif not k.startswith("model") and not k.startswith("model_ema"):
            keys.remove(k)
    for mk in reversed(m_keys):
        if not mk.startswith("unet") and not mk.startswith("unet_ema"):
            continue
    keys.remove("model_ema.num_updates")
    keys.append("model_ema.num_updates")
    for k, mk in zip(keys, m_keys):
        v = d[k]
        mv = md[mk]
        assert v.shape == mv.shape, f"{k} ({v.shape}) != {mk} ({mv.shape})"
        nd[mk] = v
        key_mapping_inv[mk] = k
    return nd


def _convert(d: tensor_dict_type, md: tensor_dict_type) -> tensor_dict_type:
    key_mapping_inv.clear()
    d, md = map(shallow_copy_dict, [d, md])
    nd = {}
    d = {k: v.cpu() for k, v in d.items()}
    md = {k: v.cpu() for k, v in md.items()}
    if "cond_stage_model.channel_mapper.weight" in d:
        print(">  [info] injecting channel_mapper")
        assert (
            d["cond_stage_model.channel_mapper.weight"].shape
            == md["condition_model.channel_mapper.weight"].shape
        )
        k = "cond_stage_model.channel_mapper.weight"
        mk = "condition_model.channel_mapper.weight"
        nd[mk] = d[k]
        key_mapping_inv[mk] = k
    # condition
    cond_d = {}
    for k in list(d.keys()):
        if k.startswith("cond_stage_model"):
            cond_d[k] = d.pop(k)
    cond_md = {k: v for k, v in md.items() if k.startswith("condition_model")}
    for k in cond_md:
        md.pop(k)
    nd.update(_convert_cond_stage(cond_d, cond_md))
    # first stage
    fd = {k: v for k, v in d.items() if k.startswith("first_stage_model")}
    fmd = {k: v for k, v in md.items() if k.startswith("first_stage")}
    nd.update(_convert_first_stage(fd, fmd))
    # others
    od = {k: v for k, v in d.items() if not k.startswith("first_stage_model")}
    omd = {k: v for k, v in md.items() if not k.startswith("first_stage")}
    nd.update(_convert_others(od, omd))
    # ema
    ema_num_updates_k = "unet_ema.num_updates"
    if ema_num_updates_k in nd:
        nd.pop(ema_num_updates_k)
        key_mapping_inv.pop(ema_num_updates_k)
        normal_keys = [k for k in list(nd) if k.startswith("unet.")]
        ema_keys = [k for k in list(nd) if k.startswith("unet_ema")]
        for k, ema_k in zip(normal_keys, ema_keys):
            assert nd.pop(k).shape == nd[ema_k].shape
        for k, ema_k in zip(normal_keys, ema_keys):
            nd[k] = nd.pop(ema_k)
            key_mapping_inv[k] = key_mapping_inv.pop(ema_k)
    return nd


def _convert_with_key_mapping(
    d: tensor_dict_type,
    md: tensor_dict_type,
) -> tensor_dict_type:
    d, md = map(shallow_copy_dict, [d, md])
    with open(download_static("sd_mapping", extension="json"), "r") as f:
        mapping = json.load(f)
    nd = _convert_cond_stage(d, md)
    for k, mk in mapping.items():
        nd[mk] = d[k]
    for k, v in md.items():
        nd.setdefault(k, v)
    return nd


def _get_inp(
    inp: Union[str, tensor_dict_type],
    vae_inp: Optional[Union[str, tensor_dict_type]],
) -> tensor_dict_type:
    inp = get_tensors(inp)
    if vae_inp is not None:
        vae_inp = get_tensors(vae_inp)
        num_injected = 0
        for k, v in vae_inp.items():
            inp_k = f"first_stage_model.{k}"
            if inp_k in inp:
                inp[inp_k] = v
                num_injected += 1
        print(f">> injected vae parameters: {num_injected}")
    return inp


def convert(
    inp: Union[str, tensor_dict_type],
    api: DiffusionAPI,
    *,
    vae_inp: Optional[Union[str, tensor_dict_type]] = None,
    load: bool = False,
    use_mapping: bool = True,
) -> tensor_dict_type:
    inp = _get_inp(inp, vae_inp)
    with api.load_context() as m:
        md = m.state_dict()
        if not use_mapping:
            nd = _convert(inp, md)
        else:
            nd = _convert_with_key_mapping(inp, md)
        if load:
            m.load_state_dict(nd)
    return nd


def convert_v2(
    inp: Union[str, tensor_dict_type],
    api: DiffusionAPI,
    *,
    vae_inp: Optional[Union[str, tensor_dict_type]] = None,
    load: bool = False,
) -> tensor_dict_type:
    inp = _get_inp(inp, vae_inp)
    for k in [
        "log_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod",
        "sqrt_recipm1_alphas_cumprod",
    ]:
        inp.pop(k)
    with api.load_context() as m:
        md = m.state_dict()
        nd = {}
        for k in [
            "condition_model.m.text_projection.bias",
            "condition_model.m.text_transformer.attention_mask",
        ]:
            nd[k] = md.pop(k)
        with open(download_static("sd_v2_mapping", extension="json"), "r") as f:
            mapping = json.load(f)
        for k, mk in mapping.items():
            v = inp[k]
            if mk == "condition_model.m.text_projection.weight":
                v = v.t()
            nd[mk] = v.view(md[mk].shape)
        if load:
            m.load_state_dict(nd)
    return nd


def inject(inp: Union[str, tensor_dict_type], api: DiffusionAPI) -> None:
    d = get_tensors(inp)
    with api.load_context() as m:
        m.load_state_dict(d)


def convert_controlnet(inp: Union[str, tensor_dict_type]) -> tensor_dict_type:
    inp = get_tensors(inp)
    with open(download_static("sd_controlnet_mapping", extension="json"), "r") as f:
        mapping = json.load(f)
    nd = {}
    for k, v in inp.items():
        if "control_model" in k:
            nd[mapping[k]] = v
    return nd


__all__ = [
    "convert",
    "convert_v2",
    "inject",
    "convert_controlnet",
]
