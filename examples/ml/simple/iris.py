import cflearn

import numpy as np

from cfdata.tabular import TabularDataset
from cflearn.misc.toolkit import get_latest_workplace


metrics = ["acc", "auc"]
x, y = TabularDataset.iris().xy
data = cflearn.ml.MLData(x, y, is_classification=True)
m = cflearn.ml.SimplePipeline(output_dim=3, metric_names=metrics)
m.fit(data)

idata = cflearn.ml.MLInferenceData(x, y)
cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

predictions = m.predict(idata)[cflearn.PREDICTIONS_KEY]
m.save("iris")
m2 = cflearn.ml.SimplePipeline.load("iris")
assert np.allclose(predictions, m2.predict(idata)[cflearn.PREDICTIONS_KEY])
m.to_onnx("iris_onnx")
m3 = cflearn.ml.SimplePipeline.from_onnx("iris_onnx")
assert np.allclose(predictions, m3.predict(idata)[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
workplace = get_latest_workplace("_logs")
assert workplace is not None
packed_path = cflearn.ml.SimplePipeline.pack(workplace)
m4 = cflearn.ml.SimplePipeline.load(packed_path)
assert np.allclose(predictions, m4.predict(idata)[cflearn.PREDICTIONS_KEY])
