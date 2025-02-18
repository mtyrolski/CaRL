import os
import pathlib
import pickle
import sys

import hydra
import torch
from carl.utils.loggers import NeptuneCaRLLogger
from carl.utils.result_loggers import SubgoalSearchResultLogger
from dotenv import load_dotenv
from loguru import logger
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from omegaconf import OmegaConf
from transformers.integrations import NeptuneCallback

from carl.utils.loggers import NeptuneCaRLLogger

DOTENV_PATH = './.tokens.env'

logger.info('Ensuring pickle.HIGHEST_PROTOCOL 5')
pickle.HIGHEST_PROTOCOL = 5


def run(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    load_dotenv(DOTENV_PATH, override=True)
    logger.info(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    logger.info(f'NEPTUNE_API_TOKEN: {os.environ.get("NEPTUNE_API_TOKEN")}')

    # Check NEPTUNE_API_TOKEN
    if os.environ.get('NEPTUNE_API_TOKEN') is None:
        logger.error('NEPTUNE_API_TOKEN is not set')
        sys.exit(1)

    algorithm = hydra.utils.instantiate(config.algorithm)
    logger.info(f'Registered algorithm: {algorithm}')
    logger.add(sink=lambda msg: print(msg, end=''), level='INFO')

    # TODO: remove this hack
    # TODO: this is so ugly. Future me, please forgive me.

    # for eval
    if hasattr(algorithm, 'result_logger'):
        if isinstance(algorithm.result_logger, SubgoalSearchResultLogger):
            conf_to_log = OmegaConf.to_container(config)
            algorithm.result_logger.custom_logger.run['parameters'] = stringify_unsupported(conf_to_log)
            algorithm.result_logger.custom_logger.run['experiment_path'] = stringify_unsupported({
                'pwd': os.getcwd(),
                'real_pwd': os.environ.get('NEPTUNEPWD')
            })

    # for training
    if hasattr(algorithm, 'custom_logger'):
        if isinstance(algorithm.custom_logger, NeptuneCaRLLogger):
            conf_to_log = OmegaConf.to_container(config)
            algorithm.custom_logger.run['parameters'] = stringify_unsupported(conf_to_log)
            algorithm.custom_logger.run['experiment_path'] = stringify_unsupported({
                'pwd': os.getcwd(),
                'real_pwd': os.environ.get('NEPTUNEPWD')
            })

    # rl loop
    if hasattr(algorithm, 'neptune_logger'):
        if isinstance(algorithm.neptune_logger, NeptuneCallback):
            conf_to_log = OmegaConf.to_container(config)
            algorithm.neptune_logger.run['parameters'] = stringify_unsupported(conf_to_log)
            algorithm.neptune_logger.run['experiment_path'] = stringify_unsupported({
                'pwd': os.getcwd(),
                'real_pwd': os.environ.get('NEPTUNEPWD')
            })

    if config.get('float32_matmul_precision', None) is not None:
        logger.info(f'Setting float32_matmul_precision to {config.float32_matmul_precision}')
        torch.set_float32_matmul_precision(config.float32_matmul_precision)

    logger.remove()
    logger.add(sys.stderr, level='INFO')
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
