# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import argparse
import importlib

from fair_clustering.experiments import ExperimentConfig, ExperimentRunner


def load_config(name: str):
    module = importlib.import_module(f"configs.{name}")
    return module.CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="+", required=True)
    args = parser.parse_args()

    for name in args.config:
        config_dict = load_config(name)
        config = ExperimentConfig(**config_dict)
        ExperimentRunner(config).run_experiment()


if __name__ == "__main__":
    main()
