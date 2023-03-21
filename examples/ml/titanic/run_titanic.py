# type: ignore

import os
import cflearn

import numpy as np

from cftool.misc import get_latest_workspace
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import seed_everything


seed_everything(123)

file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
test_file = os.path.join(file_folder, "test.csv")

processor_config = cflearn.MLBundledProcessorConfig(label_names=["Survived"])
data = cflearn.MLData.init(processor_config=processor_config).fit(train_file)

config = cflearn.MLConfig(
    model_name="wnd",
    model_config=dict(input_dim=data.num_features, output_dim=1),
    loss_name="bce",
    metric_names=["acc", "auc"],
    lr=0.1,
    optimizer_name="sgd",
    optimizer_config=dict(nesterov=True, momentum=0.9),
    global_encoder_settings=cflearn.MLGlobalEncoderSettings(embedding_dim=8),
)
m = cflearn.api.fit_ml(data, config=config, debug=check_is_ci())

loader = data.build_loader(test_file)
results = m.predict(loader)
predictions = results[cflearn.PREDICTIONS_KEY]

export_folder = "titanic"
cflearn.api.save(m, export_folder, compress=True)
m2 = cflearn.api.load_inference(export_folder)
loader = m2.data.build_loader(test_file)
results = m2.predict(loader)
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY])

latest = get_latest_workspace("_logs")
assert latest is not None
cflearn.api.pack(latest, export_folder)
m3 = cflearn.api.load_inference(export_folder)
loader = m3.data.build_loader(test_file)
results = m3.predict(loader)
assert np.allclose(predictions, results[cflearn.PREDICTIONS_KEY])
results = m3.predict(loader, return_probabilities=True)
probabilities = results[cflearn.PREDICTIONS_KEY]
assert np.allclose(probabilities.sum(1), np.ones(probabilities.shape[0]))
results = m3.predict(loader, return_classes=True)
classes = results[cflearn.PREDICTIONS_KEY]

with open(test_file, "r") as f:
    f.readline()
    id_list = [line.strip().split(",")[0] for line in f]
with open("submission.csv", "w") as f:
    f.write("PassengerId,Survived\n")
    for test_id, c in zip(id_list, classes):
        f.write(f"{test_id},{c.item()}\n")
