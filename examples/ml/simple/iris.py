import cflearn

import numpy as np

from cfdata.tabular import TabularDataset


metrics = ["acc", "auc"]
x, y = TabularDataset.iris().xy
m = cflearn.ml.SimplePipeline(
    output_dim=3,
    is_classification=True,
    loss_name="focal",
    metric_names=metrics,
)
m.fit(x, y)
cflearn.ml.evaluate(x, y, metrics=metrics, pipelines=m)

predictions = m.predict(x)[cflearn.PREDICTIONS_KEY]
m.save("iris")
m2 = cflearn.ml.SimplePipeline.load("iris")
assert np.allclose(predictions, m2.predict(x)[cflearn.PREDICTIONS_KEY])
m.to_onnx("iris_onnx")
m3 = cflearn.ml.SimplePipeline.from_onnx("iris_onnx")
assert np.allclose(predictions, m3.predict(x)[cflearn.PREDICTIONS_KEY], atol=1.0e-5)
