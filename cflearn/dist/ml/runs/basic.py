import os
import cflearn

from ._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    data = info.data
    assert data is not None
    cuda = info.meta["cuda"]
    carefree = info.meta.get("carefree", True)
    m_base = cflearn.ml.CarefreePipeline if carefree else cflearn.ml.SimplePipeline
    m = m_base(**kwargs).fit(data, cuda=cuda)
    m.save(os.path.join(info.workplace, cflearn.ML_PIPELINE_SAVE_NAME))
