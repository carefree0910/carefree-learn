import cflearn

import numpy as np

from cfdata.tabular import TabularDataset
from cflearn.misc.toolkit import check_is_ci


metrics = ["mae", "mse"]
x, y = TabularDataset.california().xy
y = (y - y.mean()) / y.std()
m = cflearn.api.fit_ml(
    x,
    y,
    is_classification=False,
    metric_names=metrics,
    debug=check_is_ci(),
)

idata = cflearn.MLInferenceData(x, y)
cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

predictions = m.predict(idata)[cflearn.PREDICTIONS_KEY]
m.save("california")
m2 = cflearn.api.load("california")
assert np.allclose(predictions, m2.predict(idata)[cflearn.PREDICTIONS_KEY])
