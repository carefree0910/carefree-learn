import os
import cflearn

from PIL import Image
from cftool.cv import to_uint8


file_folder = os.path.dirname(__file__)
api = cflearn.cv.third_party.LaMaAPI("cpu")
out = api.inpaint(
    f"{file_folder}/assets/original.png",
    f"{file_folder}/assets/mask.png",
)
out = to_uint8(out)
Image.fromarray(out).save("out.png")
