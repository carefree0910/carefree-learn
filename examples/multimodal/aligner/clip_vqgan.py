# type: ignore

import cflearn
import argparse

# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci)


text = "a tree in the garden"

if __name__ == "__main__":
    data = cflearn.dl.DummyData()
    m = cflearn.DLZoo.load_pipeline(
        "multimodal/clip_vqgan_aligner",
        model_config={"text": text},
        fixed_steps=1 if is_ci else None,
    )
    m.fit(data, cuda=None if is_ci else 0)
