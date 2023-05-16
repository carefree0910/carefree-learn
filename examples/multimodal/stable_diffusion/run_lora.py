# type: ignore

import cflearn

from cftool.misc import timeit
from cflearn.api.cv.diffusion import SDVersions


# use some famous LoRA weights as an example
# download them and place them in the current working dir before running the following scripts
key = "line_art"
key2 = "arknights"
# https://civitai.com/api/download/models/28907
path = "animeoutlineV4_16.safetensors"
# "https://civitai.com/api/download/models/7974"
path2 = "Arknights-Texas the Omertosa.safetensors"

# here we prepared two base models: anything_v3 and an anime_hybrid_model
versions = [SDVersions.ANIME_ANYTHING, SDVersions.ANIME_ORANGE]
# we'll first use anything_v3 for experiment
version = SDVersions.ANIME_ANYTHING

# some settings
kw = dict(
    # prompt
    txt="1girl, monochrome, line art (realistic:0.2), anime illustration of (misty:1.2), yellow shrit, jean shorts, suspenders, simple background, small breasts, cute girl, young girl, sunny day, with shadows, clouds ((masterpiece)), (SFW)",
    # negative prompt
    unconditional_cond=[
        "(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy, multiple arms, 2girl, 2boy, nude, breasts, (NSFW)"
    ],
    # CFG scale
    unconditional_guidance_scale=6.5,
    # seed
    seed=123,
)

# typical txt2img
api = cflearn.cv.DiffusionAPI.from_sd("cuda:0", use_half=True)
api.prepare_sd(versions)
api.switch_sd(version)
api.txt2img(**kw, export_path="out0.png")

# use single LoRA
with timeit("load lora"):
    api.load_sd_lora(key, path=path)
with timeit("inject lora"):
    api.inject_sd_lora(key)
api.txt2img(**kw, export_path="out1.png")

# multi-LoRA is also supported
with timeit("load lora"):
    api.load_sd_lora(key2, path=path2)
# inject both of the loaded LoRA to enable multi-LoRA
with timeit("inject lora"):
    api.inject_sd_lora(key, key2)
api.txt2img(**kw, export_path="out2.png")

# it is always easy to cleanup LoRAs
with timeit("cleanup lora"):
    api.cleanup_sd_lora()
# notice that `out3.png` should be identical to `out0.png`
api.txt2img(**kw, export_path="out3.png")

# and inject them back
with timeit("inject lora"):
    api.inject_sd_lora(key, key2)
# notice that `out4.png` should be identical to `out2.png`
api.txt2img(**kw, export_path="out4.png")

# notice that both the cleanup process and the injection process are lightning fast!

# setting LoRA scales is also supported
with timeit("set lora scales"):
    api.set_sd_lora_scales({key: 0.5, key2: 1.5})
# in `out5.png`:
# * the `line_art` style should be weaker
# * the `arknights` characteristics should be stronger
api.txt2img(**kw, export_path="out5.png")

# we can also switch to another (prepared) base model (e.g. the anime_hybrid_model)
# on the fly. Notice that there will always have only one checkpoint weights loaded
# to the GPU, which means as long as your RAM is large enough, you can load & switch
# between base models very quickly!
with timeit("switch SD base model"):
    api.switch_sd(SDVersions.ANIME_ORANGE)
api.txt2img(**kw, export_path="out6.png")
