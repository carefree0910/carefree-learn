import cflearn

from ._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    data_list = info.data_list
    assert data_list is not None
    sample_weights = kwargs.pop("sample_weights", None)
    m = cflearn.make(**kwargs)
    m.fit(*data_list, sample_weights=sample_weights)
    compress = info.meta.get("compress", True)
    cflearn.save(m, saving_folder=info.workplace, compress=compress)
