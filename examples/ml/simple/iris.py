import os
import cflearn

import numpy as np

from cfdata.tabular import TabularDataset
from cflearn.misc.toolkit import get_latest_workplace


metrics = ["acc", "auc"]
x, y = TabularDataset.iris().xy
m = cflearn.api.fit_ml(x, y, is_classification=True, output_dim=3, metric_names=metrics)

idata = cflearn.MLInferenceData(x, y)
cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

p = m.predict(idata)[cflearn.PREDICTIONS_KEY]
m.save("iris")
m2 = cflearn.api.load("iris")
assert np.allclose(p, m2.predict(idata)[cflearn.PREDICTIONS_KEY])
m.to_onnx("iris_onnx", onnx_only=False)
m3 = cflearn.ml.SimplePipeline.from_onnx("iris_onnx")
assert np.allclose(p, m3.predict(idata)[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
workplace = get_latest_workplace("_logs")
assert workplace is not None
packed_path = cflearn.api.pack(workplace)
m4 = cflearn.api.load(packed_path)
assert np.allclose(p, m4.predict(idata)[cflearn.PREDICTIONS_KEY])

m = cflearn.api.fit_ml(x, y, is_classification=True, output_dim=3, metric_names=metrics)
p2 = m.predict(idata)[cflearn.PREDICTIONS_KEY]
packed_paths = []
for stuff in sorted(os.listdir("_logs"))[-2:]:
    folder = os.path.join("_logs", stuff)
    packed_paths.append(cflearn.api.pack(folder))
fused = cflearn.ml.SimplePipeline.fuse_multiple(packed_paths)
p_fused = fused.predict(idata)[cflearn.PREDICTIONS_KEY]
assert np.allclose(0.5 * (p + p2), p_fused)
