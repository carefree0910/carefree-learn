import os
import cflearn
import argparse

from cflearn import MLData


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    file_folder = os.path.dirname(__file__)
    train_file = os.path.join(file_folder, "train.csv")
    test_file = os.path.join(file_folder, "test.csv")
    data = MLData.with_cf_data(train_file, cf_data_config={"label_name": "Survived"})
    m = cflearn.ml.CarefreePipeline()
    m.fit(data, cuda=args.local_rank)
