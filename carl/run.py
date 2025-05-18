import os
import pickle
import sys

import hydra
import torch
from carl.algorithms.algorithm import Algorithm
from carl.utils.result_loggers import SubgoalSearchResultLogger
from dotenv import load_dotenv
from loguru import logger
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from omegaconf import OmegaConf


HIGHEST_PROTOCOL = 5
pickle.HIGHEST_PROTOCOL = HIGHEST_PROTOCOL
DOTENV_PATH = './.tokens.env'
logger.info(F'Ensuring pickle.HIGHEST_PROTOCOL is set to {HIGHEST_PROTOCOL}')

def handle_logging(algorithm: Algorithm, config: OmegaConf, logger_key_name: str = 'result_logger'):
    pwd = os.getcwd()
    real_pwd = os.environ.get('NEPTUNEPWD')
    if not hasattr(algorithm, logger_key_name):
        logger.info(f'No {logger_key_name} found in algorithm, skipping logging to external services.')
        return
    
    assert isinstance(getattr(algorithm, logger_key_name), SubgoalSearchResultLogger), \
        f'Expected {logger_key_name} to be of type NeptuneCaRLLogger, but got {type(getattr(algorithm, logger_key_name))}'

    conf_to_log = OmegaConf.to_container(config)
    algorithm.result_logger.custom_logger.run['parameters'] = stringify_unsupported(conf_to_log)
    algorithm.result_logger.custom_logger.run['experiment_path'] = stringify_unsupported({
        'pwd': pwd,
        'real_pwd': real_pwd,
    })
    
def handle_precision(algorithm: Algorithm, config: OmegaConf):
    if config.get('float32_matmul_precision', None) is not None:
        logger.info(f'Setting float32_matmul_precision to {config.float32_matmul_precision}')
        torch.set_float32_matmul_precision(config.float32_matmul_precision)

            
def run(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    load_dotenv(DOTENV_PATH, override=True)
    logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    logger.info(f'NEPTUNE_API_TOKEN: {os.environ.get("NEPTUNE_API_TOKEN")}')
    logger.remove()
    logger.add(sys.stderr, level='INFO')
    # logger.add(sink=lambda msg: print(msg, end=''), level='INFO')
    # Check NEPTUNE_API_TOKEN
    if os.environ.get('NEPTUNE_API_TOKEN') is None:
        logger.error('NEPTUNE_API_TOKEN is not set')
        sys.exit(1)

    algorithm = hydra.utils.instantiate(config.algorithm)
    logger.info(f'Registered algorithm: {algorithm}')
    
    handle_logging(algorithm, config)
    handle_precision(algorithm, config)

    logger.info('Setting recursion limit to 2147483640')
    sys.setrecursionlimit(2147483640)

    algorithm.run()



# pylint: disable=missing-function-docstring
@hydra.main(version_base=None, config_path=None)
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
