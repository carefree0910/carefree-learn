import torch
import cflearn

import numpy as np

# for reproduction
np.random.seed(142857)
torch.manual_seed(142857)

# preparation
data_config = {"label_name": "Survived"}


def write_submissions(name: str, predictions_: np.ndarray) -> None:
    with open("test.csv", "r") as f:
        f.readline()
        id_list = [line.strip().split(",")[0] for line in f]
    with open(name, "w") as f:
        f.write("PassengerId,Survived\n")
        for test_id, prediction in zip(id_list, predictions_.ravel()):
            f.write(f"{test_id},{prediction}\n")


# wide and deep
m = cflearn.make("wnd", data_config=data_config)
m.fit("train.csv")

cflearn.evaluate("train.csv", pipelines=m, contains_labels=True)

predictions = m.predict("test.csv", contains_labels=False)
write_submissions("submissions.csv", predictions)  # type: ignore

# tree linear
m = cflearn.make("tree_linear", data_config=data_config).fit("train.csv")
predictions = m.predict("test.csv", contains_labels=False)
write_submissions("submissions_tree_linear.csv", predictions)  # type: ignore
