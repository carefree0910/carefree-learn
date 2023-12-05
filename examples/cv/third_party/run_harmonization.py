import os
import cflearn

from PIL import Image


file_folder = os.path.dirname(__file__)
api = cflearn.cv.third_party.ImageHarmonizationAPI("cpu")
out = api.run(f"{file_folder}/assets/original.png", f"{file_folder}/assets/mask.png")
Image.fromarray(out).save("out.png")
