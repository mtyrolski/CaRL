import tempfile

import pytest
from carl.slurm.grid_search import (CarlGrid, EmptyListError, NotInConfigError, NotListError)
from omegaconf import OmegaConf

from carl.slurm.grid_search import CarlGrid
from carl.slurm.grid_search import EmptyListError
from carl.slurm.grid_search import NotInConfigError
from carl.slurm.grid_search import NotListError

DUMMY_YAML_DATA = """
a: 14
b: 13
carl_grid:
    - a: [1, 2]
      b: [3, 4]
"""


def test_valid_config():
    config = {
        'algorithm': {
            'a': 14,
            'b': {
                'c': 21
            }
        },
        'carl_grid': [{
            'algorithm.a': [1, 2],
            'algorithm.b.c': [3, 4]
        }],
    }
    CarlGrid(config)    # This should not raise an error.


def test_invalid_not_list():
    config = {
        'algorithm': {
            'a': 14,
            'b': {
                'c': 21
            }
        },
        'carl_grid': [{
            'algorithm.a': [1, 2],
            'algorithm.b.c': 14
        }],
    }
    with pytest.raises(NotListError):
        CarlGrid(config)


def test_invalid_empty_list():
    config = {'carl_grid': [{'a': []}]}
    with pytest.raises(EmptyListError):
        CarlGrid(config)


def test_invalid_key_not_in_config():
    config = {'carl_grid': [{'invalid_key': [1, 2]}]}
    with pytest.raises(NotInConfigError):
        CarlGrid(config)


def test_from_file():
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write(DUMMY_YAML_DATA)
        f.flush()
        cg = CarlGrid.from_file(f.name)
    assert OmegaConf.create(cg.config) == {
        'a': 14,
        'b': 13,
        'carl_grid': [{
            'a': [1, 2],
            'b': [3, 4]
        }],
    }


@pytest.mark.parametrize(
    'config, expected',
    [
        ({}, []),
        (
            {
                'b': 14,
                'a': 2137,
                'c': 'test',
                'carl_grid': [{
                    'a': [1, 2],
                    'b': [3, 4]
                }],
                'carl_workers': {
                    'test_worker': {
                        'c': 'xxx'
                    }
                }
            },
            [
                {
                    'a': 1,
                    'b': 3,
                    'c': 'xxx'
                },
                {
                    'a': 1,
                    'b': 4,
                    'c': 'xxx'
                },
                {
                    'a': 2,
                    'b': 3,
                    'c': 'xxx'
                },
                {
                    'a': 2,
                    'b': 4,
                    'c': 'xxx'
                },
            ],
        ),
    ],
)
def test_iter_grid(config, expected):
    cg = CarlGrid(config)
    result = list(cg.iter_grid())

    assert len(result) == 0 or result[0]['test_worker'] == expected[0]
