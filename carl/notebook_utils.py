import functools
from carl.algorithms.algorithm import Algorithm
from carl.slurm.grid_search import CarlGrid
import subprocess
from loguru import logger


def instantiate_algorithm(config_name: str,
                          config_path: str = "experiments",
                          disable_gpu: bool = True,
                          worker_type: str | None = None,
                          n_jobs: int | None = None,
                          config_transformations: list = []) -> Algorithm:
    """
    Instantiate an algorithm based on the provided configuration.

    Args:
        config_name (str): The name of the configuration to use.
        config_path (str, optional): The path to the configuration directory. Defaults to "experiments".
        disable_gpu (bool, optional): Whether to disable GPU usage. Defaults to True.
        worker_type (str | None, optional): The type of worker to use. Defaults to None.
        n_jobs (int | None, optional): The number of jobs to use by overriding. Defaults to None.

    Returns:
        Algorithm: An instance of the instantiated algorithm.

    Raises:
        HydraException: If there is an error during Hydra initialization or configuration composition.

    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    from dotenv import load_dotenv
    load_dotenv('.tokens.env', override=True)
    if disable_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    initialize(config_path=config_path)
    config = compose(config_name=config_name)
    config = functools.reduce(lambda c, t: t(c), config_transformations, config)
    print(config)

    from hydra.utils import instantiate

    if worker_type is None:
        algo = instantiate(config.algorithm)
        return algo

    if n_jobs is not None:
        config.n_jobs = n_jobs

    worker2config = CarlGrid(config).iter_workers(config)
    worker_config = worker2config[worker_type]
    algo = instantiate(worker_config.algorithm)
    return algo


def save_current_notebook_to_html(notebook_name):
    """Saves the current notebook to an HTML file, requires the notebook's name as input."""
    try:
        command = [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            notebook_name    # The notebook file to convert
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        logger.success("Notebook successfully saved to HTML.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in saving notebook to HTML: {e.stderr}")
        return None
