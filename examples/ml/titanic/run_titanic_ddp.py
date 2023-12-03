# type: ignore

import os
import cflearn

from cftool.misc import get_latest_workspace


file_folder = os.path.dirname(__file__)
run_file = os.path.join(file_folder, "_titanic_task.py")
cflearn.api.run_accelerate(run_file, set_config=False)

latest = get_latest_workspace("_ddp")
assert latest is not None
m = cflearn.api.load_inference(os.path.join(latest, "pipeline"))
test_file = os.path.join(file_folder, "test.csv")
loader = m.data.build_loader(test_file)
classes = m.predict(loader, return_classes=True)[cflearn.PREDICTIONS_KEY]
with open(test_file, "r") as f:
    f.readline()
    id_list = [line.strip().split(",")[0] for line in f]
with open("submission.csv", "w") as f:
    f.write("PassengerId,Survived\n")
    for test_id, c in zip(id_list, classes):
        f.write(f"{test_id},{c.item()}\n")
