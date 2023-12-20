import cflearn
import platform

from cflearn.toolkit import check_is_ci


if __name__ == "__main__":
    is_ci = check_is_ci()
    num_jobs = 0 if platform.system() == "Linux" and is_ci else 2
    num_multiple = max(1, num_jobs) if is_ci else 5
    cflearn.api.run_multiple(
        "single_run.py",
        tag="test_run_multiple",
        cuda_list=None,
        num_jobs=num_jobs,
        num_multiple=num_multiple,
    )
