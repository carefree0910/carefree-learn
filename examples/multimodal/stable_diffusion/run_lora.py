import cflearn


key = "Any name you like"
path = "Path to the LoRA"

api = cflearn.cv.DiffusionAPI.from_sd("cuda:0", use_half=True)
api.load_sd_lora(key, path=path)
api.inject_sd_lora(key)
api.txt2img("1girl, masterpiece, best quality", "out.png")
