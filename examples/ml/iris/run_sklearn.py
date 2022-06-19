import os
import pickle

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from cflearn.dist.ml.runs._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    # data
    data = info.data
    assert data is not None
    x, y = data.x_train, data.y_train
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # model
    model = kwargs["core_name"]
    if model == "decision_tree":
        base = DecisionTreeClassifier
    elif model == "random_forest":
        base = RandomForestClassifier
    else:
        raise NotImplementedError
    sk_model = base()
    # train & save
    sk_model.fit(x, y.ravel())
    with open(os.path.join(info.workplace, "sk_model.pkl"), "wb") as f:
        pickle.dump(sk_model, f)
