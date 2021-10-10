import cflearn

import numpy as np

from cfdata.tabular import TabularDataset


base = cflearn.ml.SimplePipeline
metrics = ["mae", "mse"]
x, y = TabularDataset.boston().xy
y = (y - y.mean()) / y.std()
data = cflearn.MLData(x, y, is_classification=False)
m = cflearn.api.make("ml.simple", config={"metric_names": metrics}).fit(data)
assert isinstance(m, cflearn.ml.SimplePipeline)

idata = cflearn.MLInferenceData(x, y)
cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

predictions = m.predict(idata)[cflearn.PREDICTIONS_KEY]
m.save("boston")
m2 = base.load("boston")
assert np.allclose(predictions, m2.predict(idata)[cflearn.PREDICTIONS_KEY])
m.to_onnx("boston_onnx")
m3 = base.from_onnx("boston_onnx")
assert np.allclose(predictions, m3.predict(idata)[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
