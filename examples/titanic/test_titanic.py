import os
import torch
import cflearn

from cfdata.tabular import TabularData


file_folder = os.path.dirname(__file__)
train_file = os.path.join(file_folder, "train.csv")
test_file = os.path.join(file_folder, "test.csv")
data = TabularData(label_name="Survived").read(train_file)

train_data = cflearn.MLData(*data.processed.xy)
test_data = cflearn.MLData(*data.transform(test_file, contains_labels=False).xy)

train_loader = cflearn.MLLoader(train_data, shuffle=True)
test_loader = cflearn.MLLoader(test_data, shuffle=False)

loss = cflearn.CrossEntropyLoss()
model = cflearn.FCNN(data.processed_dim, data.num_classes, [64, 64])
inference = cflearn.MLInference(model)
trainer = cflearn.Trainer(metrics=cflearn.Accuracy())
trainer.fit(loss, model, inference, train_loader)

# print(inference.get_outputs(torch.device("cpu"), test_loader))
