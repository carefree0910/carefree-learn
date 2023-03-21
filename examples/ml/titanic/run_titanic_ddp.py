import os
import cflearn


file_folder = os.path.dirname(__file__)
run_file = os.path.join(file_folder, "_titanic_task.py")
cflearn.api.run_ddp(run_file, [1, 3])
