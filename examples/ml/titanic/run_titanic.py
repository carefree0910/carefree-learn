import os
import cflearn

import numpy as np

from cflearn.misc.toolkit import get_latest_workplace


base = cflearn.ml.CarefreePipeline
file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
test_file = os.path.join(file_folder, "test.csv")
data = cflearn.MLData.with_cf_data(
    train_file,
    cf_data_config={"label_name": "Survived"},
)
m = base().fit(data)
assert isinstance(m, base)

idata = cflearn.MLInferenceData(test_file)
results = m.predict(idata, make_loader_kwargs={"contains_labels": False})
predictions = results[cflearn.PREDICTIONS_KEY]

export_folder = "titanic"
m.save(export_folder)
m2 = base.load(export_folder)
results = m2.predict(idata, make_loader_kwargs={"contains_labels": False})
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY])

latest = get_latest_workplace("_logs")
assert latest is not None
m3 = base.load(base.pack(latest))
results = m3.predict(idata, make_loader_kwargs={"contains_labels": False})
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY])

onnx_folder = "titanic_onnx"
m.to_onnx(onnx_folder)
m4 = cflearn.ml.CarefreePipeline.from_onnx(onnx_folder)
results = m4.predict(idata, make_loader_kwargs={"contains_labels": False})
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
