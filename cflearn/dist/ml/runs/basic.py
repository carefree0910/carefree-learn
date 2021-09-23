import os
import cflearn

from ._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    data_list = info.data_list
    assert data_list is not None
    cf_data_config = kwargs.pop("cf_data_config", {})
    data = cflearn.ml.MLData.with_cf_data(*data_list, cf_data_config=cf_data_config)
    m = cflearn.ml.CarefreePipeline(**kwargs).fit(data, cuda=info.meta["cuda"])
    m.save(os.path.join(info.workplace, cflearn.ML_PIPELINE_SAVE_NAME))
