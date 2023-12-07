import cflearn

from cflearn.data.blocks import *
from cftool.misc import get_latest_workspace
from cflearn.toolkit import check_is_ci


is_ci = check_is_ci()

data_config = cflearn.TorchDataConfig(shuffle_train=False)
data_config.batch_size = 4 if is_ci else 16
data_config.shuffle_train = False
processor_config = cflearn.DataProcessorConfig()
processor_config.set_blocks(
    TupleToBatchBlock(),
    ToNumpyBlock(),
    AffineNormalizeBlock(127.5, 127.5),
    HWCToCHWBlock(),
)
images = cflearn.mnist_data(data_config, processor_config)

workplace = "_logs"
vqvae_log_folder = get_latest_workspace(workplace)
config = cflearn.DLConfig(
    model="ar",
    module_name="pixel_cnn",
    module_config=dict(num_codes=32, in_channels=1, num_classes=10),
    loss_name="cross_entropy",
    loss_config=dict(is_auto_regression=True),
    monitor_names=["mean_std", "plateau"],
)
if is_ci:
    config.to_debug()
    config.callback_names = []
inference = cflearn.cv.VQVAEInference(
    config,
    workspace=workplace,
    vqvae_log_folder=vqvae_log_folder,
    num_classes=10,
    device=None if is_ci else 0,
)
data_config = cflearn.DataConfig(batch_size=4 if is_ci else 16)
inference.fit(images, data_config)
