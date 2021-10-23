import cflearn

import numpy as np

from cfdata.tabular import TabularDataset


metrics = ["mae", "mse"]
x, y = TabularDataset.boston().xy
y = (y - y.mean()) / y.std()
m = cflearn.api.fit_ml(x, y, is_classification=False, metric_names=metrics)

idata = cflearn.MLInferenceData(x, y)
cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

predictions = m.predict(idata)[cflearn.PREDICTIONS_KEY]
m.save("boston")
m2 = cflearn.api.load("boston")
assert np.allclose(predictions, m2.predict(idata)[cflearn.PREDICTIONS_KEY])
m.to_onnx("boston_onnx")
m3 = cflearn.ml.SimplePipeline.from_onnx("boston_onnx")
assert np.allclose(predictions, m3.predict(idata)[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
