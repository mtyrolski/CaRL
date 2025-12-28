"""
Utilities for creating and iterating over parameter grids in Carl configs,
supporting worker-specific overrides.
"""

import copy
import itertools
from typing import Any, Dict, Generator, cast

import yaml
from loguru import logger
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from carl.utils.loggers import log_error_and_raise


class NotListError(Exception):
    pass


class EmptyListError(Exception):
    pass


class NotInConfigError(Exception):
    pass


class CarlGrid:
    """Manage parameter grids and produce flattened configuration dictionaries."""
    grid_literal: str = 'carl_grid'

    def __init__(self, config: dict[str, Any]):
        self.config = config
        CarlGrid.validate_config(self.config)

    @staticmethod
    def _has_nested_key(config: DictConfig, dot_key: str) -> bool:
        """Check if a nested key (dot separated) exists in the config."""
        try:
            value = OmegaConf.select(config, dot_key, throw_on_missing=True)
            return value is not None
        except Exception:
            return False

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> None:
        """Ensure 'carl_grid' entries are valid lists, non-empty, and keys exist in config."""
        # skip if no grid defined
        if cls.grid_literal not in config:
            logger.debug(f'No {cls.grid_literal} found in config. Skipping validation.')
            return

        c = 0
        logger.info(f'Validating {cls.grid_literal} syntax.')
        # Remove carl_grid from config for key checking
        config_for_keys = {k: v for k, v in config.items() if k != cls.grid_literal}
        config_omega = OmegaConf.create(config_for_keys)
        for cartesian_entry in config[cls.grid_literal]:
            logger.info(f'Validating cartesian entry: {cartesian_entry}')
            for key, value in cartesian_entry.items():
                c += 1
                if not isinstance(value, (list, ListConfig)):
                    log_error_and_raise(
                        f'All values of {cls.grid_literal} must be lists. Got {value} of key {key}',
                        exception_cls=NotListError
                    )

                if len(value) == 0:
                    log_error_and_raise(
                        f'All lists of {cls.grid_literal} must have at least one element. Got {value} of key {key}',
                        exception_cls=EmptyListError
                    )

                # Use improved nested key checking
                logger.info(f'Validating key {key} in config {list(config_for_keys.keys())}')
                if not cls._has_nested_key(config_omega, key):
                    log_error_and_raise(
                        f'All keys of {cls.grid_literal} must be inside the config. Got {key}, keys are {list(config_for_keys.keys())}',
                        exception_cls=NotInConfigError
                    )
        logger.success(f'Validated {c} entries of {cls.grid_literal}. Syntax is OK.')

    @classmethod
    def from_file(cls, path: str) -> 'CarlGrid':
        """Load a YAML file at the given path and instantiate CarlGrid."""
        # read config from YAML
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def __len__(self) -> int:
        return len(list(self.iter_grid()))

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """Alias for iter_grid to make the grid directly iterable."""
        return self.iter_grid()

    def iter_workers(self, config: Dict[str, Any] | DictConfig) -> Dict[str, Dict[str, Any]]:
        """Expand 'carl_workers' overrides into per-worker configuration dicts."""
        config_copy = OmegaConf.create(copy.deepcopy(config))
        assert isinstance(config_copy, DictConfig)
        worker2overrides = config_copy['carl_workers']
        workername2config_dict: dict[str, dict[str, Any]] = {}
        for worker_name, overrides in worker2overrides.items():
            config_copy = OmegaConf.create(copy.deepcopy(config))
            assert isinstance(config_copy, DictConfig)
            if 'carl_workers' in config_copy:
                del config_copy['carl_workers']
            # Apply overrides using OmegaConf.from_dotlist
            for key, value in overrides.items():
                config_copy = cast(DictConfig, OmegaConf.merge(config_copy, OmegaConf.from_dotlist([f'{key}={value}'])))
            assert isinstance(config_copy, DictConfig)
            workername2config_dict[worker_name] = cast(
                dict[str, Any],
                OmegaConf.to_container(config_copy, resolve=True),
            )
        return workername2config_dict

    def iter_grid(self) -> Generator[Dict[str, Any], None, None]:
        """Yield configurations by iterating over the Cartesian product defined under 'carl_grid'."""
        # if no grid key, yield nothing
        if self.grid_literal not in self.config:
            return

        for cartesian_entry in self.config[self.grid_literal]:
            all_value_combinations = itertools.product(*cartesian_entry.values())
            for values in all_value_combinations:
                config_copy = OmegaConf.create(copy.deepcopy(self.config))
                assert isinstance(config_copy, DictConfig)
                for key, value in zip(cartesian_entry.keys(), values):
                    config_copy = cast(DictConfig, OmegaConf.merge(
                        config_copy,
                        OmegaConf.from_dotlist([f'{key}={value}']),
                    ))
                assert isinstance(config_copy, DictConfig)
                if self.grid_literal in config_copy:
                    del config_copy[self.grid_literal]
                if 'carl_workers' in config_copy:
                    yield self.iter_workers(config_copy)
                else:
                    yield cast(dict[str, Any], OmegaConf.to_container(config_copy, resolve=True))

    def iter_grid_without_workers(self) -> Generator[Dict[str, Any], None, None]:
        """Yield flattened configurations without applying any worker overrides."""
        if self.grid_literal not in self.config:
            yield self.config
            return

        for cartesian_entry in self.config[self.grid_literal]:
            all_value_combinations = itertools.product(*cartesian_entry.values())
            for values in all_value_combinations:
                config_copy = OmegaConf.create(copy.deepcopy(self.config))
                assert isinstance(config_copy, DictConfig)
                for key, value in zip(cartesian_entry.keys(), values):
                    config_copy = cast(DictConfig, OmegaConf.merge(
                        config_copy,
                        OmegaConf.from_dotlist([f'{key}={value}']),
                    ))
                assert isinstance(config_copy, DictConfig)
                if self.grid_literal in config_copy:
                    del config_copy[self.grid_literal]
                yield cast(dict[str, Any], OmegaConf.to_container(config_copy, resolve=True))
