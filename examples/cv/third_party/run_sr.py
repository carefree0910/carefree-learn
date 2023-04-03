import os
import cflearn


file_folder = os.path.dirname(__file__)
api = cflearn.cv.TranslatorAPI.from_esr("cpu")
api.sr(f"{file_folder}/assets/original.png", "out.png")
