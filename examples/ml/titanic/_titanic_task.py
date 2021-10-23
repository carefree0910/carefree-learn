import os
import cflearn


if __name__ == "__main__":
    file_folder = os.path.dirname(__file__)
    train_file = os.path.join(file_folder, "train.csv")
    kwargs = dict(carefree=True, cf_data_config={"label_name": "Survived"})
    cflearn.api.fit_ml(train_file, fixed_epoch=100, **kwargs)
