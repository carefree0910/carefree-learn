import cflearn

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from cflearn.toolkit import check_is_ci
from cflearn.toolkit import seed_everything


seed_everything(123)

x = np.random.random([1000, 10])
w = np.random.random([10, 1])
y = x.dot(w) * 100.0
config = cflearn.DLConfig(
    module_name="fcnn",
    module_config=dict(input_dim=x.shape[1], output_dim=y.shape[1]),
    loss_name="multi_task",
    loss_config=dict(loss_names=["mae", "mse"]),
)


def make_kw(block_names: List[str]) -> Dict[str, Any]:
    return dict(
        config=config,
        processor_config=cflearn.MLAdvancedProcessorConfig(block_names),
        debug=check_is_ci(),
    )


def compare() -> None:
    predictions = m.predict(loader)[cflearn.PREDICTIONS_KEY]
    print("> compare\n", np.hstack([predictions[:5], y[:5]]))


m = cflearn.api.fit_ml(x, y, **make_kw([]))
loader = m.data.build_loader(x, y)
metrics = m.evaluate(loader)
compare()

m = cflearn.api.fit_ml(x, y, **make_kw(["ml_preprocessor"]))
loader = m.data.build_loader(x, y)
metrics2 = m.evaluate(loader)
compare()
print("> metrics ", metrics)
print("> metrics2", metrics2)

if check_is_ci():
    assert metrics2.final_score > metrics.final_score
