# type: ignore

import os
import shutil
import cflearn

import numpy as np


file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
tmp_folder = os.path.join(file_folder, ".tmp")
os.makedirs(tmp_folder)
with open(train_file, "r") as f:
    train_data = []
    for i, line in enumerate(f):
        if i == 0:
            train_data.append(line.split(","))
            continue
        l = line.find('"')
        r = line.rfind('"')
        ld = line[:l].split(",")[:-1]
        rd = line[r:].split(",")[1:]
        name = line[l : r + 1]
        train_data.append(ld + [name] + rd)
column_names = train_data[0].copy()
column_names[-1] = column_names[-1].strip()


def make_test_file(label_name: str) -> str:
    label_idx = column_names.index(label_name)
    output_path = os.path.join(tmp_folder, f"{label_name}.csv")
    with open(output_path, "w") as f:
        for line in train_data:
            line = line.copy()
            line.pop(label_idx)
            joined = ",".join(line)
            if label_idx != len(column_names) - 1:
                f.write(joined)
            else:
                f.write(joined + "\n")
    return output_path


def check(label_name: str):
    test_file = make_test_file(label_name)
    kwargs = dict(
        debug=True,
        carefree=True,
        cf_data_config={"label_name": label_name},
    )
    m: cflearn.ml.MLCarefreePipeline
    m = cflearn.api.fit_ml(train_file, x_valid=train_file, **kwargs)
    idata = m.make_inference_data(test_file, contains_labels=False)
    pred1 = m.predict(idata, return_classes=True)[cflearn.PREDICTIONS_KEY]
    m.data.save("./data")
    m.save("./tmp")
    m2: cflearn.ml.MLCarefreePipeline = cflearn.api.load("./tmp")
    idata = m2.make_inference_data(test_file, contains_labels=False)
    pred2 = m2.predict(idata, return_classes=True)[cflearn.PREDICTIONS_KEY]
    assert np.all(pred1 == pred2)

    data = cflearn.DLDataModule.load("./data")
    m3 = cflearn.ml.MLCarefreePipeline(config=cflearn.MLConfig(fixed_steps=1))
    m3.fit(data)
    idata = m3.make_inference_data(test_file, contains_labels=False)
    pred3 = m2.predict(idata, return_classes=True)[cflearn.PREDICTIONS_KEY]
    assert np.all(pred2 == pred3)

    print(pred3[:10].ravel())


if __name__ == "__main__":
    check("Survived")
    check("Sex")
    check("Pclass")
    check("Embarked")
    check("SibSp")
    os.remove("tmp.zip")
    shutil.rmtree("./data")
    shutil.rmtree(tmp_folder)
