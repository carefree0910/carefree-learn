import os
import cflearn

import numpy as np

from cflearn.misc.toolkit import get_latest_workplace


file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
test_file = os.path.join(file_folder, "test.csv")
kwargs = dict(carefree=True, cf_data_config={"label_name": "Survived"})
m = cflearn.api.fit_ml(train_file, **kwargs)  # type: ignore

idata = cflearn.MLInferenceData(test_file)
results = m.predict(idata, make_loader_kwargs={"contains_labels": False})
predictions = results[cflearn.PREDICTIONS_KEY]

export_folder = "titanic"
m.save(export_folder)
m2 = cflearn.api.load(export_folder)
results = m2.predict(idata, make_loader_kwargs={"contains_labels": False})
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY])

latest = get_latest_workplace("_logs")
assert latest is not None
m3 = cflearn.api.load(cflearn.api.pack(latest))
results = m3.predict(idata, make_loader_kwargs={"contains_labels": False})
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY])

onnx_folder = "titanic_onnx"
m.to_onnx(onnx_folder)
m4 = cflearn.ml.CarefreePipeline.from_onnx(onnx_folder)
results = m4.predict(idata, make_loader_kwargs={"contains_labels": False})
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
