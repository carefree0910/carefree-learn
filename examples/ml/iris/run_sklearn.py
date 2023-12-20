import os
import pickle

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from cflearn.constants import INPUT_KEY
from cflearn.constants import LABEL_KEY
from cflearn.dist.ml.runs._utils import get_info


if __name__ == "__main__":
    info = get_info()
    meta = info.meta
    # data
    data = info.data
    assert data is not None
    loader = data.get_loaders()[0]
    dataset = loader.get_full_batch()
    x, y = dataset[INPUT_KEY], dataset[LABEL_KEY]
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # model
    model = meta["module"]
    if model == "decision_tree":
        base = DecisionTreeClassifier
    elif model == "random_forest":
        base = RandomForestClassifier
    else:
        raise NotImplementedError
    sk_model = base()
    # train & save
    sk_model.fit(x, y.ravel())
    with open(os.path.join(info.workspace, "sk_model.pkl"), "wb") as f:
        pickle.dump(sk_model, f)
