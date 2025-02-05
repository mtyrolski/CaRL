import os
import sys
from functools import partial
from typing import Any
from uuid import uuid4

import joblib as jl
from loguru import logger


def increase_limit():
    recursion_limit = 2147483640
    curr_recursion_limit = sys.getrecursionlimit()

    if curr_recursion_limit < recursion_limit:
        sys.setrecursionlimit(recursion_limit)


NOT_READY_LABEL = 'not_ready'
stop_signal = '.carl_stop_signal'


def match_label(f: str, label) -> bool:
    return f.startswith(label)


def load_experiences(f):
    increase_limit()
    try:
        return jl.load(f)
    except:
        logger.error(f'Error while loading {f}.')
        return None


def get_latest_file(file_paths):
    if not file_paths:
        return None

    # Get the file with the maximum modification time
    return max(file_paths, key=os.path.getmtime)


def dump_resource(resource: Any, label: str):    # type: ignore
    """Dumps a resource to a file."""
    short_uuid = str(uuid4())[:8]
    filename = f'{label}_{short_uuid}.jl'
    tmp_filename = f'{NOT_READY_LABEL}_{filename}'
    jl.dump(resource, tmp_filename)

    # Rename file
    # Note: this trick is for avoiding reading not ready resources
    os.rename(tmp_filename, filename)
    logger.debug(f'Dumped resource {label} to {filename}')


def read_resource_and_delete(
    label: str,
    flatten: bool = True,
    limit_resources_to_read: int | None = None,
    parallel: bool = False,
) -> Any:
    """Reads a resource from a file and deletes it.

    It is not thread safe.
    """
    fs = list(filter(partial(match_label, label=label), os.listdir('.')))

    if not parallel:
        objs = [jl.load(f) for f in fs]
    else:
        objs = jl.Parallel(n_jobs=-1)(jl.delayed(load_experiences)(f) for f in fs)

    # Limit resources to read
    if limit_resources_to_read is not None:
        objs = objs[:limit_resources_to_read]
        fs = fs[:limit_resources_to_read]

    # Delete files
    for f in fs:
        os.remove(f)

    if flatten:
        objs = [obj for sublist in objs for obj in sublist]

    logger.debug(f'Read and deleted {len(objs)} resources with label {label}')
    return objs


def read_resource(label: str,
                  path='.',
                  parallel: bool = True,
                  n_jobs: int | None = None,
                  limit: int | None = None) -> Any:
    """Reads a resource from a file."""
    fs = [os.path.join(path, f) for f in filter(partial(match_label, label=label), os.listdir(path))]

    logger.info(f'Found {len(fs)} resources with label {label}')

    if limit is not None:
        logger.info(f'Limiting resources to read to {limit}')
        fs = fs[:limit]

    if not parallel:
        objs = [jl.load(f) for f in fs]
    else:
        print('parallel')
        n_jobs = n_jobs if n_jobs is not None else 10
        objs = jl.Parallel(n_jobs=n_jobs, verbose=1)(jl.delayed(load_experiences)(f) for f in fs)

    # Filtering out None's so we skip incorrect files
    objs = list(filter(lambda x: x is not None, objs))
    objs = [obj for sublist in objs for obj in sublist]

    return objs


def exists_resource(label: str) -> bool:
    """Checks if a resource exists."""

    logger.debug(f'Checking if resource {label} exists')

    fs = list(filter(match_label, os.listdir('.')))

    logger.debug(f'Resource {label} exists: {len(fs) > 0}, {fs}')

    return len(fs) > 0
