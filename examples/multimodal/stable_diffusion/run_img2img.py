import os
import cflearn


file_folder = os.path.dirname(__file__)
api = cflearn.multimodal.DiffusionAPI.from_sd(device="cuda:0", use_half=True)
api.img2img(
    f"{file_folder}/assets/cat.png",
    "out.png",
    cond=["A lovely dog."],
    fidelity=0.1,
    seed=234,
)
