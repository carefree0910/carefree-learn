import cflearn


api = cflearn.cv.DiffusionAPI.from_sd("cuda:0", use_half=True)
api.txt2img("A lovely little cat.", "out.png", seed=123)
