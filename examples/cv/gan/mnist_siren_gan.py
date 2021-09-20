# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

data = cflearn.cv.MNISTData(
    root="../data",
    batch_size=4 if is_ci else 64,
    transform="for_generation",
)

m = cflearn.DLZoo.load_pipeline("gan/siren.gray", img_size=28, debug=is_ci)
m.fit(data, cuda=None if is_ci else 0)
