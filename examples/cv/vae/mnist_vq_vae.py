# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

num_classes = 10
data = cflearn.cv.MNISTData(batch_size=4 if is_ci else 64, transform="for_generation")

m = cflearn.DLZoo.load_pipeline(
    "vae/vq.gray_lite",
    img_size=28,
    model_config={"num_classes": num_classes},
    callback_configs={"vq_vae": {"num_classes": num_classes}},
    debug=is_ci,
)
m.fit(data, cuda=None if is_ci else 1)
