import torch

from torch import nn
from typing import Union
from cftool.misc import shallow_copy_dict
from cftool.array import tensor_dict_type

from ..api.cv.models.diffusion import DiffusionAPI


def _convert_first_stage(d: tensor_dict_type, md: tensor_dict_type) -> tensor_dict_type:
    d, md = map(shallow_copy_dict, [d, md])
    keys = list(d.keys())
    m_keys = list(md.keys())
    for k in [
        "core.first_stage.core.codebook.embedding.weight",
        "core.first_stage.core.to_embedding.weight",
        "core.first_stage.core.to_embedding.bias",
        "core.first_stage.core.from_embedding.weight",
        "core.first_stage.core.from_embedding.bias",
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
    before = keys[:start_idx]
    up = keys[start_idx:end_idx]
    after = keys[end_idx:]
    num_up = int(keys[end_idx - 1].split(".")[3]) + 1
    new_up = []
    mapping = shallow_copy_dict(d)
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
    keys = before + new_up + after

    def _handle_attn(prefix):
        attn_keys = {}
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
        if not mk.startswith("core.unet") and not mk.startswith("core.unet_ema"):
            nd[mk] = md[mk]
            m_keys.remove(mk)
    keys.remove("model_ema.num_updates")
    keys.append("model_ema.num_updates")
    for k, mk in zip(keys, m_keys):
        v = d[k]
        mv = md[mk]
        assert v.shape == mv.shape, f"{k} ({v.shape}) != {mk} ({mv.shape})"
        nd[mk] = v
    return nd


def _convert(d: tensor_dict_type, md: tensor_dict_type) -> tensor_dict_type:
    d, md = map(shallow_copy_dict, [d, md])
    nd = {}
    d = {k: v.cpu() for k, v in d.items()}
    md = {k: v.cpu() for k, v in md.items()}
    if "cond_stage_model.channel_mapper.weight" in d:
        print(">  [info] injecting channel_mapper")
        assert (
            d["cond_stage_model.channel_mapper.weight"].shape
            == md["core.condition_model.channel_mapper.weight"].shape
        )
        nd["core.condition_model.channel_mapper.weight"] = d[
            "cond_stage_model.channel_mapper.weight"
        ]
    # condition
    for k in list(d.keys()):
        if k.startswith("cond_stage_model"):
            d.pop(k)
    cond_md = {k: v for k, v in md.items() if k.startswith("core.condition_model")}
    for k in cond_md:
        md.pop(k)
    # first stage
    fd = {k: v for k, v in d.items() if k.startswith("first_stage_model")}
    fmd = {k: v for k, v in md.items() if k.startswith("core.first_stage")}
    nd.update(_convert_first_stage(fd, fmd))
    # others
    od = {k: v for k, v in d.items() if not k.startswith("first_stage_model")}
    omd = {k: v for k, v in md.items() if not k.startswith("core.first_stage")}
    nd.update(_convert_others(od, omd))
    # ema
    if "core.unet_ema.num_updates" in nd:
        nd.pop("core.unet_ema.num_updates")
        normal_keys = [k for k in list(nd) if k.startswith("core.unet.")]
        ema_keys = [k for k in list(nd) if k.startswith("core.unet_ema")]
        for k, ema_k in zip(normal_keys, ema_keys):
            assert nd.pop(k).shape == nd[ema_k].shape
        for k, ema_k in zip(normal_keys, ema_keys):
            nd[k] = nd.pop(ema_k)
    # condition
    for k, v in cond_md.items():
        nd.setdefault(k, v)
    return nd


def convert(
    inp: Union[str, tensor_dict_type],
    api: DiffusionAPI,
    *,
    load: bool = False,
) -> tensor_dict_type:
    class Wrapper(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.core = m

    if isinstance(inp, str):
        inp = torch.load(inp, map_location="cpu")
        if "state_dict" in inp:
            inp = inp["state_dict"]
    api_cond, m_cond = api.cond_model, api.m.condition_model
    api.m.condition_model = api_cond
    wrapper = Wrapper(api.m)
    nd = _convert(inp, wrapper.state_dict())
    if load:
        wrapper.load_state_dict(nd)
    api.m.condition_model = m_cond
    return nd


__all__ = [
    "convert",
]
