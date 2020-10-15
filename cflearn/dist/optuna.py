import argparse

import cflearn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda")
    parser.add_argument("--num_trial")
    parser.add_argument("--config")
    parser.add_argument("--key_mapping")
    cflearn.optuna_core(parser.parse_args())
