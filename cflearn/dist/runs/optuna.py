import cflearn

from cflearn.api.hpo import OptunaArgs

from ._utils import get_info


if __name__ == "__main__":
    info = get_info(requires_data=False)
    cflearn.optuna_core(
        OptunaArgs(
            info.meta["cuda"],
            info.meta["compress"],
            info.meta["num_trial"],
            info.meta["task_config_folder"],
            info.meta["key_mapping_folder"],
        )
    )
