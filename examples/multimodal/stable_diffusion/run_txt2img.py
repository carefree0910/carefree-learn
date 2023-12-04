import cflearn


api = cflearn.multimodal.DiffusionAPI.from_sd(device="cuda:0", use_half=True)
api.txt2img("A lovely little cat.", "out.png", seed=123)
