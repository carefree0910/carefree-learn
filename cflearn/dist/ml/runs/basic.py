import cflearn

from ._utils import get_info


if __name__ == "__main__":
    info = get_info()
    data = info.data
    assert data is not None
    cuda = info.meta["cuda"]
    assert info.config is not None
    cflearn.MLTrainingPipeline.init(info.config).fit(data, cuda=cuda)
