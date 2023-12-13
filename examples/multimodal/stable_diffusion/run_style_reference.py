import cflearn


api = cflearn.multimodal.DiffusionAPI.from_sd(device="cuda:0", use_half=True)
api.setup_hooks(
    style_reference_image="assets/cat.png",
    style_reference_states=dict(
        style_fidelity=0.5,
        reference_weight=1.0,
    ),
)
api.txt2img("A lovely little dog.", "out.png", seed=123)
