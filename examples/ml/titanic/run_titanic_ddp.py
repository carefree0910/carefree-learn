import os
import cflearn


if __name__ == "__main__":
    file_folder = os.path.dirname(__file__)
    train_file = os.path.join(file_folder, "train.csv")
    test_file = os.path.join(file_folder, "test.csv")
    m = cflearn.ml.CarefreePipeline(data_config={"label_name": "Survived"})
    m.ddp(train_file, cuda_list=[1, 2])
