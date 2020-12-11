import cflearn

from cflearn.api.hpo import OptunaArgs

from ._utils import get_info


if __name__ == "__main__":
    info = get_info(requires_data=False)
    meta, config = info.meta, info.kwargs
    cflearn.optuna_core(
        OptunaArgs(
            meta["cuda"],
            meta["num_trial"],
            meta["task_config_folder"],
            meta["key_mapping_folder"],
        )
    )
