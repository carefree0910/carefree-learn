import cflearn

from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import get_latest_workplace


is_ci = check_is_ci()

data = cflearn.cv.MNISTData(shuffle=False, transform="for_generation")
workplace = "_logs" if is_ci else "c_vq_vae_inference_v4"
vqvae_log_folder = get_latest_workplace(workplace) if is_ci else "c_vq_vae_v4"
config = dict(
    num_codes=16,
    vqvae_log_folder=vqvae_log_folder,  # type: ignore
    num_classes=10,
    cuda=None if is_ci else 1,
    debug=is_ci,
)
if is_ci:
    config["callback_names"] = []
inference = cflearn.cv.VQVAEInference(workplace, **config)
inference.fit(data)
