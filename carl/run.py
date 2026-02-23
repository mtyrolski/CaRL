'''\
Entry point for running experiments configured via Hydra and CarlGrid.
Handles environment setup, logging, precision settings, and dispatch of algorithm runs.
'''  # module docstring

import os
import pickle
import sys
from typing import Any, Final

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
from carl.utils.secrets import mask_secret

HIGHEST_PROTOCOL: Final[int] = pickle.HIGHEST_PROTOCOL
DOTENV_PATH = './.tokens.env'
logger.info(f'Using pickle.HIGHEST_PROTOCOL={HIGHEST_PROTOCOL}')

def handle_logging(
    algorithm: Algorithm,
    config: DictConfig,
    logger_key_name: str = 'result_logger'
) -> None:
    """
    Register experiment parameters and paths with the algorithm's external logger, if available.
    """
    pwd = os.getcwd()  # current working directory
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
    
def handle_precision(config: DictConfig) -> None:
    """
    Adjust torch float32 matmul precision if specified in the config.
    """
    if config.get('float32_matmul_precision', None) is not None:
        logger.info(f'Setting float32_matmul_precision to {config.float32_matmul_precision}')
        torch.set_float32_matmul_precision(config.float32_matmul_precision)

def _instantiate_and_run(exp_config: DictConfig) -> None:
    """
    Instantiate the Algorithm via Hydra and execute its run() method,
    applying logging and recursion limit settings.
    """
    algorithm: Algorithm = hydra.utils.instantiate(exp_config.algorithm)  # type: ignore[assignment]
    assert isinstance(algorithm, Algorithm), \
        f'Expected algorithm to be of type Algorithm, but got {type(algorithm)}'
    logger.info(f'Registered algorithm: {algorithm}')
    
    handle_logging(algorithm, exp_config)
    handle_precision(exp_config)
    _RECURSION_LIMIT: Final[int] = 2147483640
    if sys.getrecursionlimit() < _RECURSION_LIMIT:
        logger.warning(f'Raising recursion limit from {sys.getrecursionlimit()} to {_RECURSION_LIMIT}')
        sys.setrecursionlimit(_RECURSION_LIMIT)
    algorithm.run()

   
def run(config: DictConfig) -> None:
    """
    Main runner: log config, load environment vars, verify tokens,
    and execute all experiments in the CarlGrid.
    """
    logger.info(OmegaConf.to_yaml(config))  # dump config to logs
    load_dotenv(DOTENV_PATH, override=True)
    neptune_api_token = os.environ.get('NEPTUNE_API_TOKEN')
    logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    logger.info(f'NEPTUNE_API_TOKEN: {mask_secret(neptune_api_token)}')
    logger.remove()
    logger.add(sys.stderr, level='INFO')

    if neptune_api_token is None:
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


@hydra.main(version_base=None, config_path=None)
def main(config: DictConfig) -> None:
    """
    Hydra entry point that calls run() with parsed DictConfig.
    """
    run(config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
