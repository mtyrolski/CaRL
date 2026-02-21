"""
Resource utilities: dumping, loading, and managing joblib-based resources with logging.
"""

import os
import sys
from typing import Any, List, Optional
from uuid import uuid4

import joblib as jl
from loguru import logger


def increase_limit() -> None:
    """Increase recursion limit to a high threshold if not already set."""
    target_limit = 2_147_483_640
    current = sys.getrecursionlimit()
    if current < target_limit:
        sys.setrecursionlimit(target_limit)


NOT_READY_LABEL = 'not_ready'
stop_signal = '.carl_stop_signal'


def match_label(filename: str, label: str) -> bool:
    """Check if filename starts with the given label."""
    return filename.startswith(label)


def load_experiences(file_path: str) -> Optional[Any]:
    """Load a joblib file, catching errors and returning None on failure."""
    increase_limit()
    try:
        return jl.load(file_path)
    except Exception as e:
        logger.error(f'Error loading {file_path}: {e}')
        return None


def get_latest_file(file_paths: List[str]) -> Optional[str]:
    """Return the most recently modified file from a list, or None if empty."""
    if not file_paths:
        return None
    return max(file_paths, key=os.path.getmtime)


def dump_resource(resource: Any, label: str) -> None:
    """Atomically dump an object to a joblib file with a unique label."""
    short_id = uuid4().hex[:8]
    filename = f'{label}_{short_id}.jl'
    tmp_name = f'{NOT_READY_LABEL}_{filename}'
    jl.dump(resource, tmp_name)
    os.replace(tmp_name, filename)
    logger.debug(f'Dumped resource {label} to {filename}')


def read_resource_and_delete(
    label: str,
    flatten: bool = True,
    limit_resources_to_read: Optional[int] = None,
    parallel: bool = False,
) -> List[Any]:
    """Reads resources matching label, deletes files, and returns list of objects."""
    files = [f for f in os.listdir('.') if match_label(f, label)]

    if not parallel:
        objs = [jl.load(f) for f in files]
    else:
        objs = list(jl.Parallel(n_jobs=-1)(jl.delayed(load_experiences)(f) for f in files))

    if limit_resources_to_read is not None:
        objs = objs[:limit_resources_to_read]
        files = files[:limit_resources_to_read]

    for f in files:
        os.remove(f)

    if flatten:
        objs = [item for sub in objs for item in sub]  # type: ignore

    logger.debug(f'Read and deleted {len(objs)} resources labeled {label}')
    return objs  # type: ignore


def read_resource(
    label: str,
    path: str = '.',
    parallel: bool = True,
    n_jobs: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Any]:
    """Load and return resources matching label from a directory, optionally in parallel."""
    files = [os.path.join(path, f) for f in os.listdir(path) if match_label(f, label)]

    logger.info(f'Found {len(files)} resources with label {label}')

    if limit is not None:
        logger.info(f'Limiting resources to read to {limit}')
        files = files[:limit]

    if not parallel:
        objs = [jl.load(f) for f in files]
    else:
        workers = n_jobs or 10
        logger.info(f'Loading resources in parallel with {workers} jobs')
        objs = jl.Parallel(n_jobs=workers, verbose=1)(jl.delayed(load_experiences)(f) for f in files)

    # filter out failures and flatten
    valid = [o for o in objs if o is not None]
    return [item for sub in valid for item in (sub if isinstance(sub, list) else [sub])]


def exists_resource(label: str) -> bool:
    """Checks if a resource exists with the given label in current directory."""

    logger.debug(f'Checking if resources start with label: {label}')

    files = [f for f in os.listdir('.') if match_label(f, label)]

    logger.debug(f'Found {len(files)} files with label: {label}')

    return bool(files)
