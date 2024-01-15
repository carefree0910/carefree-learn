# type: ignore

import os
import cflearn

from cflearn.toolkit import check_is_ci
from cflearn.toolkit import seed_everything


seed_everything(123)

file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
test_file = os.path.join(file_folder, "test.csv")

processor_config = cflearn.MLBundledProcessorConfig(
    label_names=["Survived"],
    custom_dtypes=dict(SibSp=cflearn.DataTypes.FLOAT, Parch=cflearn.DataTypes.FLOAT),
)
data = cflearn.MLData.init(processor_config=processor_config).fit(train_file)

config = cflearn.MLConfig(
    module_name="wnd",
    module_config=dict(input_dim=data.num_features, output_dim=1),
    loss_name="bce",
    metric_names=["acc", "auc"],
    lr=0.1,
    optimizer_name="sgd",
    optimizer_config=dict(nesterov=True, momentum=0.9),
    encoder_settings={"3": cflearn.MLEncoderSettings(2, methods="one_hot")},
    global_encoder_settings=cflearn.MLGlobalEncoderSettings(embedding_dim=8),
)
m = cflearn.api.fit_ml(data, config=config, debug=check_is_ci())

interpreter = cflearn.ml.Interpreter(data, m.build_model.model)
interpreter.interpret(test_file, export_path="titanic_interpret.png")
