import os
import cflearn

from cftool.misc import get_latest_workspace


file_folder = os.path.dirname(__file__)
run_file = os.path.join(file_folder, "_titanic_task.py")
cflearn.api.run_accelerate(run_file, set_config=False)
