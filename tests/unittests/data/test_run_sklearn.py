import os
import pickle

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from cflearn.dist.runs._utils import get_info

if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    data_list = info.data_list
    assert data_list is not None
    x, y = data_list[:2]
    model = kwargs["model"]
    sk_model = (SVR if model == "svr" else LinearSVR)()
    sk_model.fit(x, y.ravel())
    with open(os.path.join(info.workplace, "sk_model.pkl"), "wb") as f:
        pickle.dump(sk_model, f)
