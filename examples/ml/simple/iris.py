import os
import cflearn

import numpy as np

from cfdata.tabular import TabularDataset
from cflearn.misc.toolkit import get_latest_workplace


base = cflearn.ml.SimplePipeline
metrics = ["acc", "auc"]
x, y = TabularDataset.iris().xy
data = cflearn.ml.MLData(x, y, is_classification=True)
m = base(output_dim=3, metric_names=metrics).fit(data)
assert isinstance(m, base)

idata = cflearn.ml.MLInferenceData(x, y)
cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

p = m.predict(idata)[cflearn.PREDICTIONS_KEY]
m.save("iris")
m2 = base.load("iris")
assert np.allclose(p, m2.predict(idata)[cflearn.PREDICTIONS_KEY])
m.to_onnx("iris_onnx")
m3 = base.from_onnx("iris_onnx")
assert np.allclose(p, m3.predict(idata)[cflearn.PREDICTIONS_KEY], atol=1.0e-4)
workplace = get_latest_workplace("_logs")
assert workplace is not None
packed_path = base.pack(workplace)
m4 = base.load(packed_path)
assert np.allclose(p, m4.predict(idata)[cflearn.PREDICTIONS_KEY])

m = base(output_dim=3, metric_names=metrics).fit(data)
p2 = m.predict(idata)[cflearn.PREDICTIONS_KEY]
packed_paths = []
for stuff in sorted(os.listdir("_logs"))[-2:]:
    folder = os.path.join("_logs", stuff)
    packed_paths.append(base.pack(folder))
fused = base.fuse_multiple(packed_paths)
p_fused = fused.predict(idata)[cflearn.PREDICTIONS_KEY]
assert np.allclose(0.5 * (p + p2), p_fused)
