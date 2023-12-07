# type: ignore

import os
import shutil
import cflearn

import numpy as np

from cftool.misc import get_latest_workspace
from cflearn.toolkit import check_is_ci
from cflearn.toolkit import seed_everything
from cflearn.data.ml import iris_dataset


is_ci = check_is_ci()
seed_everything(123)

metrics = ["acc", "auc"]
x, y = iris_dataset()
config = cflearn.DLConfig(
    module_name="fcnn",
    module_config=dict(input_dim=x.shape[1], output_dim=3),
    loss_name="focal",
    metric_names=metrics,
)
if is_ci and os.path.isdir("_logs"):
    shutil.rmtree("_logs")

block_names = ["ml_recognizer", "ml_preprocessor"]
kw = dict(
    config=config,
    processor_config=cflearn.MLAdvancedProcessorConfig(block_names),
    debug=is_ci,
)
m = cflearn.api.fit_ml(x, y, **kw)
loader = m.data.build_loader(x, y)
print("> metrics", m.evaluate(loader))

p = m.predict(loader)[cflearn.PREDICTIONS_KEY]
cflearn.api.save(m, "iris", compress=True)
m2 = cflearn.api.load_inference("iris")
loader = m2.data.build_loader(x, y)
assert np.allclose(p, m2.predict(loader)[cflearn.PREDICTIONS_KEY])
m3 = cflearn.DLInferencePipeline.build_with(
    m2.config, m2.build_model.model.state_dict()
)
assert np.allclose(p, m3.predict(loader)[cflearn.PREDICTIONS_KEY])

cflearn.api.pack(get_latest_workspace("_logs"), "iris")
m3 = cflearn.api.load_inference("iris")
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

m = cflearn.api.fit_ml(x, y, **kw)
cflearn.api.save(m, "iris", compress=True)
m2 = cflearn.api.load_training("iris")
m2.fit(m.data)
loader = m2.data.build_loader(x, y)
metrics = m.evaluate(loader)
metrics2 = m2.evaluate(loader)
print("> metrics ", metrics)
print("> metrics2", metrics2)
cflearn.api.evaluate(loader, dict(m1=m, m2=m2))

if is_ci:
    assert metrics.final_score < metrics2.final_score
