import cflearn

from cftool.misc import get_latest_workplace
from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

data = cflearn.cv.MNISTData(shuffle=False, transform="for_generation")
workplace = "_logs"
vqvae_log_folder = get_latest_workplace(workplace)
config = dict(
    num_codes=16,
    vqvae_log_folder=vqvae_log_folder,  # type: ignore
    num_classes=10,
    cuda=None if is_ci else 1,
    debug=is_ci,
)
if is_ci:
    config["callback_names"] = []  # type: ignore
inference = cflearn.cv.VQVAEInference(workplace, **config)  # type: ignore
inference.fit(data)
