import cflearn

import numpy as np

from cfdata.tabular import TabularDataset
from cflearn.misc.toolkit import get_latest_workplace


metrics = ["acc", "auc"]
x, y = TabularDataset.iris().xy
input_dim = x.shape[1]
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
workplace = get_latest_workplace("_logs")
packed_path = cflearn.ml.SimplePipeline.pack(workplace, input_dim=input_dim)
m4 = cflearn.ml.SimplePipeline.load(packed_path)
assert np.allclose(predictions, m4.predict(x)[cflearn.PREDICTIONS_KEY])
