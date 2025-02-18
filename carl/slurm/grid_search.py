import copy
import itertools
from typing import Any

import yaml
from loguru import logger
from omegaconf import ListConfig
from omegaconf import OmegaConf


class NotListError(Exception):
    pass


class EmptyListError(Exception):
    pass


class NotInConfigError(Exception):
    pass


class CarlGrid:
    grid_literal: str = 'carl_grid'

    def __init__(self, config: dict[str, Any]):
        self.config = config
        CarlGrid.validate_config(self.config)

    @classmethod
    def validate_config(cls, config):
        if cls.grid_literal not in config:
            logger.debug(f'No {cls.grid_literal} found in config. Skipping validation.')
            return
        c = 0
        logger.info(f'Validating {cls.grid_literal} syntax.')
        for cartesian_entry in config[cls.grid_literal]:
            for key, value in cartesian_entry.items():
                c += 1
                # check that all values are lists
                if not isinstance(value, list | ListConfig):
                    logger.error(f'All values of {cls.grid_literal} must be lists. Got {value} of key {key}')
                    raise NotListError(f'All values of {cls.grid_literal} must be lists. Got {value} of key {key}')

                # check that each list has at least one element
                if len(value) == 0:
                    logger.error(
                        f'All lists of {cls.grid_literal} must have at least one element. Got {value} of key {key}')
                    raise EmptyListError(
                        f'All lists of {cls.grid_literal} must have at least one element. Got {value} of key {key}')

                # check that all keys are inside a config
                config_omega = OmegaConf.create(config)
                res = OmegaConf.select(config_omega, key, throw_on_resolution_failure=True)

                if res is None:
                    logger.error(
                        f'All keys of {cls.grid_literal} must be inside the config. Got {key}, keys are {list(config.keys())}'
                    )
                    raise NotInConfigError(
                        f'All keys of {cls.grid_literal} must be inside the config. Got {key}, keys are {list(config.keys())}'
                    )
        logger.success(f'Validated {c} entries of {cls.grid_literal}. Syntax is OK.')

    @classmethod
    def from_file(cls, path: str) -> 'CarlGrid':
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def __len__(self) -> int:
        return len(list(self.iter_grid()))

    def __iter__(self):
        return self.iter_grid()

    def iter_workers(self, config: dict[str, Any]):
        worker2overrides = config['carl_workers']
        # TODO: add default worker
        workername2config_dict = {}
        for worker_name, overrides in worker2overrides.items():
            config_copy = copy.deepcopy(config)
            OmegaConf.set_struct(config_copy, False)
            del config_copy['carl_workers']
            OmegaConf.set_struct(config_copy, True)
            # Override worker and add worker options
            for key, value in overrides.items():
                config_copy = OmegaConf.merge(config_copy, OmegaConf.from_dotlist([f'{key}={value}']))

            workername2config_dict[worker_name] = config_copy

        return workername2config_dict

    def iter_grid(self):
        if self.grid_literal not in self.config:
            return [self.config]

        for cartesian_entry in self.config[self.grid_literal]:
            all_value_combinations = itertools.product(*cartesian_entry.values())
            for values in all_value_combinations:
                config_copy = copy.deepcopy(self.config)
                for key, value in zip(cartesian_entry.keys(), values):
                    config_copy = OmegaConf.merge(config_copy, OmegaConf.from_dotlist([f'{key}={value}']))
                del config_copy[self.grid_literal]
                yield self.iter_workers(config_copy)
