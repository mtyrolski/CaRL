import copy
import itertools
from typing import Any, Dict, Generator

import yaml
from loguru import logger
from omegaconf import DictConfig
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

    @staticmethod
    def _has_nested_key(config: DictConfig, dot_key: str) -> bool:
        """Check if a nested key (dot separated) exists in the config."""
        try:
            # OmegaConf.select throws on missing key if throw_on_missing=True
            OmegaConf.select(config, dot_key, throw_on_missing=True)
            return True
        except _:
            return False

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> None:
        if cls.grid_literal not in config:
            logger.debug(f'No {cls.grid_literal} found in config. Skipping validation.')
            return

        c = 0
        logger.info(f'Validating {cls.grid_literal} syntax.')
        config_omega = OmegaConf.create(config)
        for cartesian_entry in config[cls.grid_literal]:
            logger.info(f'Validating cartesian entry: {cartesian_entry}')
            for key, value in cartesian_entry.items():
                c += 1
                if not isinstance(value, (list, ListConfig)):
                    logger.error(f'All values of {cls.grid_literal} must be lists. Got {value} of key {key}')
                    raise NotListError(f'All values of {cls.grid_literal} must be lists. Got {value} of key {key}')

                if len(value) == 0:
                    logger.error(f'All lists of {cls.grid_literal} must have at least one element. Got {value} of key {key}')
                    raise EmptyListError(f'All lists of {cls.grid_literal} must have at least one element. Got {value} of key {key}')

                # Use improved nested key checking
                logger.info(f'Validating key {key} in config {list(config.keys())}')
                if not cls._has_nested_key(config_omega, key):
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
        workername2config_dict = {}
        for worker_name, overrides in worker2overrides.items():
            config_copy = copy.deepcopy(config)
            if 'carl_workers' in config_copy:
                del config_copy['carl_workers']
            # Apply overrides using OmegaConf.from_dotlist
            for key, value in overrides.items():
                config_copy = OmegaConf.merge(config_copy, OmegaConf.from_dotlist([f'{key}={value}']))
            workername2config_dict[worker_name] = config_copy
        return workername2config_dict

    def iter_grid(self) -> Generator[Dict[str, Any], None, None]:
        if self.grid_literal not in self.config:
            yield self.config
            return

        for cartesian_entry in self.config[self.grid_literal]:
            all_value_combinations = itertools.product(*cartesian_entry.values())
            for values in all_value_combinations:
                config_copy = copy.deepcopy(self.config)
                for key, value in zip(cartesian_entry.keys(), values):
                    config_copy = OmegaConf.merge(config_copy, OmegaConf.from_dotlist([f'{key}={value}']))
                if self.grid_literal in config_copy:
                    del config_copy[self.grid_literal]
                yield self.iter_workers(config_copy)

    def iter_grid_without_workers(self) -> Generator[Dict[str, Any], None, None]:
        if self.grid_literal not in self.config:
            yield self.config
            return

        for cartesian_entry in self.config[self.grid_literal]:
            all_value_combinations = itertools.product(*cartesian_entry.values())
            for values in all_value_combinations:
                config_copy = copy.deepcopy(self.config)
                for key, value in zip(cartesian_entry.keys(), values):
                    config_copy = OmegaConf.merge(config_copy, OmegaConf.from_dotlist([f'{key}={value}']))
                if self.grid_literal in config_copy:
                    del config_copy[self.grid_literal]
                yield config_copy