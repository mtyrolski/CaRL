import os
import pickle
import sys
from typing import Any

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from omegaconf import OmegaConf

from carl.algorithms.algorithm import Algorithm
from carl.slurm.grid_search import CarlGrid
from carl.utils.result_loggers import SubgoalSearchResultLogger

HIGHEST_PROTOCOL = 5
pickle.HIGHEST_PROTOCOL = HIGHEST_PROTOCOL
DOTENV_PATH = './.tokens.env'
logger.info(F'Ensuring pickle.HIGHEST_PROTOCOL is set to {HIGHEST_PROTOCOL}')

def handle_logging(algorithm: Algorithm, config: DictConfig, logger_key_name: str = 'result_logger'):
    pwd = os.getcwd()
    real_pwd = os.environ.get('NEPTUNEPWD')
    if not hasattr(algorithm, logger_key_name):
        logger.info(f'No {logger_key_name} found in algorithm, skipping logging to external services.')
        return
    
    assert isinstance(getattr(algorithm, logger_key_name), SubgoalSearchResultLogger), \
        f'Expected {logger_key_name} to be of type NeptuneCaRLLogger, but got {type(getattr(algorithm, logger_key_name))}'

    conf_to_log = OmegaConf.to_container(config)
    getattr(algorithm, logger_key_name).custom_logger.run['parameters'] = stringify_unsupported(conf_to_log)
    getattr(algorithm, logger_key_name).custom_logger.run['experiment_path'] = stringify_unsupported({
        'pwd': pwd,
        'real_pwd': real_pwd,
    })
    
def handle_precision(config: DictConfig):
    if config.get('float32_matmul_precision', None) is not None:
        logger.info(f'Setting float32_matmul_precision to {config.float32_matmul_precision}')
        torch.set_float32_matmul_precision(config.float32_matmul_precision)

def _instantiate_and_run(exp_config: DictConfig) -> None:
    algorithm = hydra.utils.instantiate(exp_config.algorithm)
    logger.info(f'Registered algorithm: {algorithm}')
    
    handle_logging(algorithm, exp_config)
    handle_precision(exp_config)

    logger.info('Setting recursion limit to 2147483640')
    sys.setrecursionlimit(2147483640)
    # logger.info(f'\n======\nRunning algorithm with config:\n {OmegaConf.to_yaml(exp_config)}')
    # algorithm.run()

   
def run(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    load_dotenv(DOTENV_PATH, override=True)
    logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    logger.info(f'NEPTUNE_API_TOKEN: {os.environ.get("NEPTUNE_API_TOKEN")}')
    logger.remove()
    logger.add(sys.stderr, level='INFO')

    if os.environ.get('NEPTUNE_API_TOKEN') is None:
        logger.error('NEPTUNE_API_TOKEN is not set')
        sys.exit(1)
        
    raw_data: dict[str, Any] = OmegaConf.to_container(config) # type: ignore[assignment]
    assert isinstance(raw_data, dict), 'Config must be a dictionary'
    assert all(isinstance(key, str) for key in raw_data.keys()), 'All keys in config must be strings'
    grid = CarlGrid(raw_data)
    
    for i, exp_config in enumerate(grid.iter_grid_without_workers()):
        logger.info(f'Running experiment {i + 1}/{len(grid)}')
        exp_dict_config: DictConfig = OmegaConf.create(exp_config)
        _instantiate_and_run(exp_dict_config)


# pylint: disable=missing-function-docstring
@hydra.main(version_base=None, config_path=None)
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
