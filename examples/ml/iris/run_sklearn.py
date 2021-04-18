import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from cflearn.dist.ml.runs._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    # data
    data_list = info.data_list
    assert data_list is not None
    x, y = data_list[:2]
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
