import os
import cflearn

import numpy as np


file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
test_file = os.path.join(file_folder, "test.csv")
m = cflearn.ml.MLPipeline(data_config={"label_name": "Survived"})
m.fit(train_file)
outputs = m.predict(test_file, transform_kwargs={"contains_labels": False})
predictions = outputs.forward_results["predictions"].argmax(1)

export_folder = "titanic"
m.save(export_folder)
m = cflearn.ml.MLPipeline.load(export_folder)
outputs = m.predict(test_file, transform_kwargs={"contains_labels": False})
assert np.allclose(predictions, outputs.forward_results["predictions"].argmax(1))
