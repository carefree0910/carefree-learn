import json
import argparse
import requests

import numpy as np

from cflearn.configs import _parse_config
from mlflow.utils.cli_args import HOST
from mlflow.utils.cli_args import PORT
from cflearn.misc.toolkit import parse_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--adapter", default="http")
    parser.add_argument("--config", default=None)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--export_file", required=True)
    args = parse_args(parser.parse_args())
    config = _parse_config(args.config)
    config["file"] = args.test_file
    url = f"{args.adapter}://{args.host}:{args.port}/invocations"
    response = requests.post(
        url=url,
        data=json.dumps([config]),
        headers={"Content-Type": "application/json; format=pandas-records"},
    )
    if response.status_code != 200:
        raise Exception(f"Status Code {response.status_code}. {response.text}")
    np.save(args.export_file, np.array(json.loads(response.content)))
