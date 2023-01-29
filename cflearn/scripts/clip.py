import cflearn

from typing import Dict
from typing import Union
from cftool.misc import shallow_copy_dict
from cftool.array import tensor_dict_type


def get_open_clip_config(model_name: str) -> Dict[str, Union[str, int]]:
    def _get_num_heads(sub_cfg: Dict[str, int], default: int) -> int:
        num_heads = sub_cfg.get("heads")
        if num_heads is not None:
            return num_heads
        width = sub_cfg["width"]
        head_width = sub_cfg.get("head_width")
        if head_width is not None:
            if width % head_width != 0:
                raise ValueError("`head_width` should be divided by `width`")
            return width // head_width
        return default

    try:
        import open_clip
    except:
        raise ValueError("`open_clip` should be installed for `get_open_clip_config`")
    cfg = open_clip.get_model_config(model_name)
    if cfg is None:
        raise ValueError(f"unrecognized model_name `{model_name}` occurred")
    vision_cfg = cfg["vision_cfg"]
    text_cfg = cfg["text_cfg"]
    return dict(
        latent_dim=cfg.get("embed_dim", 512),
        vision_latent_dim=vision_cfg["width"],
        vision_patch_size=vision_cfg.get("patch_size", 32),
        vision_num_heads=_get_num_heads(vision_cfg, 12),
        vision_num_layers=vision_cfg.get("layers", 12),
        vision_feedforward_activation="GELU",
        text_latent_dim=text_cfg["width"],
        text_num_heads=_get_num_heads(text_cfg, 8),
        text_num_layers=text_cfg.get("layers", 12),
        text_feedforward_activation="GELU",
    )


def convert_open_clip(model_name: str, pretrained: str) -> tensor_dict_type:
    try:
        import open_clip
    except:
        raise ValueError("`open_clip` should be installed for `convert_open_clip`")
    cf_clip_config = get_open_clip_config(model_name)
    oc_model = open_clip.create_model(model_name, pretrained)
    d = shallow_copy_dict(oc_model.state_dict())
    cf_model = cflearn.api.clip_model(model_config=cf_clip_config, pretrained=False)
    md = shallow_copy_dict(cf_model.state_dict())
    extra = {
        "logit_scale": d.pop("logit_scale"),
        "vit.encoder.head_token": d.pop("visual.class_embedding"),
        "vit.encoder.pos_encoding.pos_encoding": d.pop("visual.positional_embedding"),
        "vit.output_projection": d.pop("visual.proj"),
        "token_embedding.weight": d.pop("token_embedding.weight"),
        "text_projection.weight": d.pop("text_projection").t(),
        "text_transformer.encoder.pos_encoding.pos_encoding": d.pop(
            "positional_embedding"
        ),
    }
    re_order_keys = [
        "text_transformer.attention_mask",
    ]
    for k, v in extra.items():
        mv = md.pop(k)
        md[k] = v.view(mv.shape)
    for k in re_order_keys:
        mv = md.pop(k)
        md[k] = mv
    m_keys = list(md)
    for i, (k, v) in enumerate(d.items()):
        mk = m_keys[i]
        mv = md[mk]
        md[mk] = v.view(mv.shape)
    return md
