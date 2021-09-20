# type: ignore

import cflearn

from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

text = "a tree in the garden"

if __name__ == "__main__":
    m = cflearn.multimodal.CLIPWithVQGANTrainer(text, fixed_steps=1 if is_ci else None)
    m.run(cuda=None if is_ci else 0)
