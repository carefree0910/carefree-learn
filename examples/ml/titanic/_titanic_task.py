import os
import cflearn

from cflearn.misc.toolkit import seed_everything


seed_everything(123)

if __name__ == "__main__":
    file_folder = os.path.dirname(__file__)
    train_file = os.path.join(file_folder, "train.csv")
    processor_config = cflearn.MLBundledProcessorConfig(label_names=["Survived"])
    data = cflearn.MLData.init(processor_config=processor_config).fit(train_file)
    config = cflearn.MLConfig(
        model_name="fcnn",
        model_config=dict(input_dim=data.num_features, output_dim=1),
        loss_name="bce",
        metric_names=["acc", "auc"],
    )
    cflearn.api.fit_ml(data, config=config)
