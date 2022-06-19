import os
import cflearn

from ._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    data = info.data
    assert data is not None
    cuda = info.meta["cuda"]
    m = cflearn.ml.CarefreePipeline(**kwargs).fit(data, cuda=cuda)
    m.save(os.path.join(info.workplace, cflearn.ML_PIPELINE_SAVE_NAME))
