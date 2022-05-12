# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

text = "a tree in the garden"

if __name__ == "__main__":
    m = cflearn.multimodal.CLIPWithVQGANAligner(text, debug=is_ci)
    m.run(cuda=None if is_ci else 0)
