import cflearn

from ._utils import get_info


if __name__ == "__main__":
    info = get_info()
    kwargs = info.kwargs
    increment_kwargs = info.increment_kwargs
    data_list = info.data_list
    assert data_list is not None
    sample_weights = info.meta.get("sample_weights")
    model = kwargs.pop("model")
    m = cflearn.make(model, kwargs, increment_kwargs)
    m.fit(*data_list, sample_weights=sample_weights)
    compress = info.meta.get("compress", True)
    cflearn.save(m, saving_folder=info.workplace, compress=compress)
