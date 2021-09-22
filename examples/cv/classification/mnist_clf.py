# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

data = cflearn.cv.MNISTData(batch_size=4 if is_ci else 64, transform="to_tensor")

m = cflearn.DLZoo.load_pipeline("clf/resnet18.gray", num_classes=10, debug=is_ci)
m.fit(data, cuda=None if is_ci else 0)
