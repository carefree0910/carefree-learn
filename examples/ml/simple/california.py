# type: ignore

import os
import cflearn

import numpy as np

from cftool.misc import get_latest_workspace
from cflearn.data.ml import california_dataset
from cflearn.toolkit import check_is_ci
from cflearn.toolkit import seed_everything


seed_everything(123)

x, y = california_dataset()
y = (y - y.mean()) / y.std()
config = cflearn.MLConfig(
    module_name="fcnn",
    module_config=dict(input_dim=x.shape[1], output_dim=1),
    loss_name="multi_task",
    loss_config=dict(loss_names=["mae", "mse"]),
)

block_names = ["ml_recognizer", "ml_preprocessor", "ml_splitter"]
kw = dict(
    config=config,
    processor_config=cflearn.MLAdvancedProcessorConfig(block_names),
    debug=check_is_ci(),
)
m = cflearn.api.fit_ml(x, y, **kw)
loader = m.data.build_loader(x, y)
print("> metrics", m.evaluate(loader))

p = m.predict(loader)[cflearn.PREDICTIONS_KEY]
cflearn.api.save(m, "california", compress=True)
m2 = cflearn.api.load_inference("california")
loader = m2.data.build_loader(x, y)
assert np.allclose(p, m2.predict(loader)[cflearn.PREDICTIONS_KEY])
cflearn.api.save(m2, "california", compress=True)
m2 = cflearn.api.load_inference("california")
loader = m2.data.build_loader(x, y)
assert np.allclose(p, m2.predict(loader)[cflearn.PREDICTIONS_KEY])
d = m2.build_model.model.state_dict()
m3 = cflearn.DLInferencePipeline.build_with(m2.config, d, data=m2.data)
assert np.allclose(p, m3.predict(loader)[cflearn.PREDICTIONS_KEY])

cflearn.api.save(m3, "california", compress=True)
m3 = cflearn.api.load_inference("california")
loader = m3.data.build_loader(x, y)
assert np.allclose(p, m3.predict(loader)[cflearn.PREDICTIONS_KEY])
cflearn.api.pack(get_latest_workspace("_logs"), "california")
m3 = cflearn.api.load_inference("california")
loader = m3.data.build_loader(x, y)
assert np.allclose(p, m3.predict(loader)[cflearn.PREDICTIONS_KEY])

m = cflearn.api.fit_ml(x, y, **kw)
loader = m.data.build_loader(x, y)
p2 = m.predict(loader)[cflearn.PREDICTIONS_KEY]
packed_paths = []
for i, stuff in enumerate(sorted(os.listdir("_logs"))[-2:]):
    folder = os.path.join("_logs", stuff)
    packed_paths.append(f"packed_{i}")
    cflearn.api.pack(folder, packed_paths[-1], pack_type=cflearn.PackType.EVALUATION)
fused = cflearn.api.fuse_inference(packed_paths)
loader = fused.data.build_loader(x, y)
p_fused = fused.predict(loader)[cflearn.PREDICTIONS_KEY]
assert np.allclose(0.5 * (p + p2), p_fused)
fused = cflearn.api.fuse_evaluation(packed_paths)
loader = fused.data.build_loader(x, y)
print("> metrics.fused", fused.evaluate(loader))
p_fused = fused.predict(loader)[cflearn.PREDICTIONS_KEY]
assert np.allclose(0.5 * (p + p2), p_fused)

cflearn.api.save(fused, "fused", compress=True)
fused2 = cflearn.api.load_inference("fused")
loader = fused2.data.build_loader(x, y)
assert np.allclose(p_fused, fused2.predict(loader)[cflearn.PREDICTIONS_KEY])
