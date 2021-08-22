import os
import cflearn

from cflearn.api.ml import MLData
from cflearn.api.ml import MLInferenceData


if __name__ == "__main__":
    file_folder = os.path.dirname(__file__)
    train_file = os.path.join(file_folder, "train.csv")
    test_file = os.path.join(file_folder, "test.csv")
    data = MLData.with_cf_data(train_file, data_config={"label_name": "Survived"})
    m = cflearn.ml.CarefreePipeline()
    m.ddp(data, cuda_list=[1, 2])
    m.predict(MLInferenceData(test_file), make_loader_kwargs={"contains_labels": False})
