# CaRL Library README

- -CaRL Library README](#carl-library-readme)
  - [Introduction](#introduction)
  - [Papers Created or Reproduced with CaRL](#papers-created-or-reproduced-with-carl)
  - [Codebase Authors](#codebase-authors)
  - [Quick Start](#quick-start)
  - [CaRL Extension Over Hydra Config](#carl-extension-over-hydra-config)
    - [Starting Point (`algorithm`)](#starting-point-algorithm)
    - [CaRL Workers (`carl_workers`)](#carl-workers-carl_workers)
    - [CaRL Grid (`carl_grid`)](#carl-grid-carl_grid)
    - [Heterogeneous Job Support](#heterogeneous-job-support)
  - [Local Execution Options](#local-execution-options)
    - [Notebook](#notebook)
    - [Command Line (cmd)](#command-line-cmd)
  - [Deploying Remote Experiments with the CaRL Launcher](#deploying-remote-experiments-with-the-carl-launcher)
    - [Image Build](#image-build)
    - [Cluster Configuration](#cluster-configuration)
    - [Examples of Running Experiments](#examples-of-running-experiments)
    - [Syncing Multiple Clusters](#syncing-multiple-clusters)
    - [Cluster Synchronization Script](#cluster-synchronization-script)
  - [Datasets Demo](#datasets-demo)
  - [Open Source Components](#open-source-components)
  - [Evaluation Results](#evaluation-results)

---

## Introduction

**Why?**  
The **CaRL (Combinatorial Reinforcement Learning) Library** is designed for developing and scaling offline and online reinforcement learning experiments. It provides a comprehensive suite of tools for planning in **combinatorial problems**, including environments, data handling, inference components, training loops, and AI-guided search algorithms. The library comes with interactive notebooks in the `examples` folder, showcasing inference components used in published research. It includes fully operational environments such as `Sokoban`, `NPuzzle`, `Rubik`, and `INT`. Additionally, we provide **35 open-source models** of various types: `Generator`, `Value`, `Conditional Low-Level Policy`, and `Policy` (see [Papers Created or Reproduced with CaRL](#papers-created-or-reproduced-with-carl) and the full list in the [Open Source Components](#open-source-components) section).

**How?**  
CaRL leverages key components of [SLURM](https://slurm.schedmd.com/documentation.html), enabling the deployment of tasks across multiple nodes with varying specifications (heterogeneous job support). It offers a flexible way to define and execute tasks using a custom deployer that extends the [Hydra Config](https://github.com/facebookresearch/hydra) syntax. Remote computation is handled via [Apptainer](https://apptainer.org/) (formerly Singularity) images, which are automatically generated. Experiment tracking is managed using [Neptune](https://neptune.ai).

---

## Papers Created or Reproduced with CaRL

- Zawalski, M., Góral, G., Tyrolski, M., Wiśnios, E., Budrowski, F., Kuciński, Ł., and Miłoś, P., 2024. *What Matters in Hierarchical Search for Combinatorial Reasoning Problems?* arXiv preprint arXiv:2406.03361.
- Zawalski, M., Tyrolski, M., Czechowski, K., Odrzygóźdź, T., Stachura, D., Piękos, P., Wu, Y., Kuciński, Ł., and Miłoś, P., 2022. *Fast and Precise: Adjusting Planning Horizon with Adaptive Subgoal Search.* arXiv preprint arXiv:2206.00702.
- Czechowski, K., Odrzygóźdź, T., Zbysiński, M., Zawalski, M., Olejnik, K., Wu, Y., Kuciński, Ł., and Miłoś, P., 2021. *Subgoal Search for Complex Reasoning Tasks.* Advances in Neural Information Processing Systems, 34, pp. 624–638.
- Selected works cited in the above papers.

---

## Codebase Authors

From late 2022 to 2024, this codebase was actively developed by various subgroups of our team. Here are the contributors and authors of the codebase:
Michał Tyrolski, Emilia Wiśnios, Gracjan Góral, Franek Budrowski, Michał Zawalski  

---

## Quick Start

1. **Set up environment variables**:  
   Ensure the following environment variables are defined in your `.tokens.env` file (see `.tokens.env.example` for reference):
   - `NEPTUNE_API_TOKEN`: Required for authentication. See [Neptune documentation](https://docs.neptune.ai/) for details.
   - `HYDRA_FULL_ERROR=1`: Enables detailed error reporting for debugging.
   - `TQDM_MININTERVAL=30`: Sets the minimum update interval for progress bars.

2. **Set up the Python environment using Poetry** (Python 3.11.4 required):
   ```bash
   cd repo_dir
   poetry shell
   poetry install
   ```

3. **Mount demonstrative data**:  
   Link the data under `./rl-data` (download and link using `ln -s`) to run notebooks from the `examples/` folder. **You can now explore the notebooks.**

4. **Run experiments from config**:  
   To execute experiments from a configuration file (required for remote or multi-node execution), follow these steps:
   - Understand the config structure: [CaRL Extension Over Hydra Config](#carl-extension-over-hydra-config).
   - Learn about [heterogeneous jobs](#heterogeneous-job-support) and how to run them remotely using the SLURM CaRL launcher.
   - Understand how to [run experiments locally](#local-execution-options) using notebooks, the command line, or [deploy them remotely](#deploying-remote-experiments-with-the-carl-launcher).

---

## CaRL Extension Over Hydra Config

The CaRL library extends the **Hydra configuration system** to provide a flexible and scalable way to define and execute reinforcement learning experiments. This extension allows users to specify complex configurations for algorithms, workers, and grid searches, which are essential for running experiments in both local and distributed environments. Below, we break down the key components of the CaRL extension using simplified examples.

### Starting Point (`algorithm`)

The `Algorithm` class is an **abstract base class** that serves as the foundation for all algorithms in the CaRL library. Every algorithm in CaRL must derive from this class and implement the `run` method, ensuring a consistent interface for all algorithms.

**Example from Code:**
```python
from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def run(self) -> None:
        pass
```

**Example Implementation:**
```python
class SolveInstances(Algorithm):
    def __init__(self, solver, data_loader, result_logger, problems_to_solve, n_parallel_workers):
        self.solver = solver
        self.data_loader = data_loader
        self.result_logger = result_logger
        self.problems_to_solve = problems_to_solve
        self.n_parallel_workers = n_parallel_workers

    def run(self) -> None:
        # Main logic for solving instances
        for problem in self.data_loader:
            result = self.solver.solve(problem)
            self.result_logger.log_results(result)
```

### CaRL Workers (`carl_workers`)

In CaRL, a **worker** is a unit of computation that performs a specific task. This is particularly useful for heterogeneous jobs, where different tasks (e.g., solving, training) are assigned to different workers.

**Example from Config:**
```yaml
carl_workers:
  loop:
    algorithm._target_: carl.algorithms.training_loop.TrainingLoopHF
  solver:
    algorithm._target_: carl.algorithms.training_loop.DistributedSolverWorker
  trainer:
    algorithm._target_: carl.algorithms.training_loop.DistributedTrainerWorker
```

For an example of deploying a multi-node job with heterogeneous workers, see the [Examples of Running Experiments](#examples-of-running-experiments) section.

### CaRL Grid (`carl_grid`)

The `carl_grid` section defines a **grid search** over hyperparameters or configurations. It parses a list of grid dictionaries and creates a Cartesian product of all combinations.

**Example from Config:**
```yaml
carl_grid:
  - algorithm.solver_class.max_nodes: [150]
    algorithm.solver_class.subgoal_generator.generator_k_list: [[8, 4, 1]]
    algorithm.solver_class.subgoal_generator.paths_to_generator_weights: [[
          "./validation/sokoban/components/full_data/generator/border/8/checkpoint-75856",
        "./validation/sokoban/components/full_data/generator/border/4/checkpoint-75856",
        "./validation/sokoban/components/full_data/generator/border/1/checkpoint-151712",
    ]]
    algorithm.solver_class.validator.cllp.path_to_conditional_low_level_policy_weights: ["./validation/sokoban/components/full_data/cllp/8/checkpoint-167585"]
  - algorithm.solver_class.max_nodes: [150]
    algorithm.solver_class.subgoal_generator.generator_k_list: [[4, 1]]
    algorithm.solver_class.subgoal_generator.paths_to_generator_weights: [[
        "./validation/sokoban/components/full_data/generator/border/4/checkpoint-75856",
        "./validation/sokoban/components/full_data/generator/border/1/checkpoint-151712",
    ]]
    algorithm.solver_class.validator.cllp.path_to_conditional_low_level_policy_weights: ["./validation/sokoban/components/full_data/cllp/4/checkpoint-167585"]
```

### Heterogeneous Job Support

CaRL supports **heterogeneous jobs**, allowing users to define multiple worker types with different configurations. This is particularly useful for experiments requiring different types of computation on different nodes or SLURM partitions (e.g., GPU nodes on a GPU partition and CPU nodes on a CPU partition).

For reference, see the following setup for a basic example of how multiple nodes can communicate with each other: [dummy config](configs/hetjob/dummy.yaml), [dummy producer](carl/algorithms/dummy_hetworkers/dummy_producer.py), [dummy receiver](carl/algorithms/dummy_hetworkers/dummy_worker.py).

CaRL also provides the following environment variables for handling heterogeneous jobs:

| Variable | Description | Default |
| --- | --- | --- |
| `CARL_SLURM_ARRAY_TASK_ID` | ID of the job within the hetgrid | 0 (always, since hetjobs do not support arrays) |
| `CARL_LOCAL_WORKER_ID` | ID of the worker within the local het group | `het_worker_idx` |
| `CARL_HET_GROUP_ID` | ID of the local het group | `het_group_idx` |

---

## Local Execution Options

### Notebook

For interactive development, you can instantiate and run algorithms directly in a Jupyter Notebook:

```python
from carl.notebook_utils import instantiate_algorithm

alg = instantiate_algorithm('rlloop_adasubs_sokoban', config_path='../experiments', worker_type='trainer')
alg.run()
```

### Command Line (cmd)

You can also run experiments via the command line:

```sh
python3 -m carl.run --config-dir experiments --config-name adaptive_subgoal_search_solve_n_puzzle
```

This method is suitable for quick tests or environments where a Jupyter Notebook is unavailable.

---

## Deploying Remote Experiments with the CaRL Launcher

CaRL includes its own implementation of a SLURM launcher. To execute an experiment, use the `launcher.py` script with the following arguments:

```sh
python3 -m carl.slurm.launcher --cluster-config CLUSTER_CONFIG --job-config JOB_CONFIG --worker WORKER
```

- `--cluster-config`: Path to the YAML file defining your cluster configuration (e.g., `carl/slurm/ares.yaml`).
- `--job-config`: Path to the experiment configuration file (e.g., `experiments/adaptive_subgoal_search_solve_n_puzzle`).
- `--worker`: Specifies the worker type and resources (e.g., `"solve;1;cpu1"` for one CPU worker of type 'solve').

### Image Build

To ensure the image is consistent with local dependencies, update the requirements file:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

Then build the image:

```bash
apptainer build carl_v0.1.0 apptainer/carl.def
```

This image can be sent to a remote server for computation.

### Cluster Configuration

The cluster configuration file specifies the resources and environment settings for the SLURM cluster. Here is an example configuration file:

```yaml
# Description: Slurm configuration for cluster
host: "hostname"
storage_dir: "/path/to/storage"
config_dir: "configurations"
repo_url: "git@github.com:username/repository.git"
data_dir: "/path/to/data"

# Apptainer exec args are used to mount directories inside the container.
apptainer_container: "/path/to/container.sif"
apptainer_exec_args:
  - "-B /path/to/local_storage:/path/to/local_storage"
  - "-B /path/to/project:/path/to/project"
  - "--env TQDM_MININTERVAL=30"

# Node specs are used to specify the resources needed for each job/or even each worker.
node_specs:
  cpu24:
    account: 'account_number'
    partition: 'partition_name'
    time: 1000
    cpus-per-task: 24
    gpus-per-task: 0
    mem-per-cpu: '5000MB'
    nodes: 1
    ntasks: 1
  gpu1:
    account: 'account_number'
    partition: 'partition_name'
    time: 1440
    cpus-per-task: 4
    gpus-per-task: 1
    mem-per-cpu: '5000MB'
    nodes: 1
    ntasks: 1
```


### Examples of Running Experiments

- **Running adaptive subgoal search on a single CPU node:**
  ```sh
  python3 -m carl.slurm.launcher --cluster-config "carl/slurm/ares.yaml" --job-config experiments/adaptive_subgoal_search_solve_n_puzzle --worker "solve;1;cpu1"
  ```

- **Running with multiple workers:**
  ```sh
  python3 -m carl.slurm.launcher --cluster-config "carl/slurm/eagle.yaml" --job-config experiments/rlloop_adasubs_n_puzzle --worker "trainer;1;cpu1" --worker "solver;5;cpu1"
  ```
  This command allocates 5 CPU nodes for 'solver' and 1 for 'trainer'.


### Syncing Multiple Clusters
Minor but useful tool for syncing data synchronization across clusters, specifying which directories to sync and ignore

Example configuration file:
```json
{
    "ignore": ["__pycache__", "venv", "expert_data"],
    "local": {"tree_root": "./path/to/local/dataset"},
    "clusters": {
        "cluster1": {
            "host_name": "hostname1",
            "tree_root": "/path/to/root/data"
        },
        "cluster2": {
            "host_name": "hostname2",
            "tree_root": "/path/to/root/data"
        },
        "cluster3": {
            "host_name": "hostname3",
            "tree_root": "/path/to/root/data"
        },
        "cluster_backup": {
            "host_name": "backup_host",
            "tree_root": "/path/to/root/data"
        },
    },
    "sync_paths": ["."]
}
```


### Cluster Synchronization Script

This script synchronizes data between clusters using rsync. It transfers data from a source cluster to multiple target clusters, ensuring directory existence and data consistency:

**Example Command:**
```bash
python sync_clusters.py -c path/to/config.json -s source_cluster -t target_cluster1 -t target_cluster2
```

equivalent to `rsync -uvar ...`.

---

## Datasets Demo

Here are some dataset samples for exploring the codebase:

- **NPuzzle:**
  - Offline trajectories: `./rl-data/validation/npuzzle/offline/basic_solver`
  - Problem instances (evaluation): `./rl-data/validation/npuzzle/progress/fin`
- **Sokoban:**
  - Offline trajectories: `./rl-data/validation/sokoban/offline/12-12-4/`
  - Problem instances: `./rl-data/validation/sokoban/progress/boards_1000_b4_gs25_c300_p0.35`, `boards_1000_b6_gs100_c300_p0.35`, `boards_1000_b7_gs100_c300_p0.35`
- **Rubik:**
  - Offline trajectories: `./rl-data/validation/rubik/offline/mixture_uniform`
  - Problem instances (evaluation): `./rl-data/validation/rubik/progress/shuffle_general`

For access to the full datasets used in various experiments, please contact the codebase authors.

---

## Open Source Components

|    | Env     | Component   | Dist   | Checkpoint                                    | Full Path                                                                   |
|---:|:--------|:------------|:-------|:----------------------------------------------|:----------------------------------------------------------------------------|
|  0 | NPuzzle | CLLP        | 4      | `cllp/4/checkpoint-294075`                    | `./rl-data/validation/npuzzle/components/moe/cllp/4/checkpoint-294075`                 |
|  1 | NPuzzle | CLLP        | 8      | `cllp/8/checkpoint-225736`                    | `./rl-data/validation/npuzzle/components/moe/cllp/8/checkpoint-225736`                 |
|  2 | NPuzzle | Generator   | 4      | `generator/4/checkpoint-48314`                | `./rl-data/validation/npuzzle/components/moe/generator/4/checkpoint-48314`             |
|  3 | NPuzzle | Generator   | 8      | `generator/8/checkpoint-64090`                | `./rl-data/validation/npuzzle/components/moe/generator/8/checkpoint-64090`             |
|  4 | NPuzzle | Policy      | N/A    | `policy/checkpoint-31552`                     | `./rl-data/validation/npuzzle/components/moe/policy/checkpoint-31552`                  |
|  5 | NPuzzle | Value       | N/A    | `value/checkpoint-2825298`                    | `./rl-data/validation/npuzzle/components/moe/value/checkpoint-2825298`                 |
|  6 | Rubik   | CLLP        | 4      | `cllp/4/checkpoint-2372409`                   | `./rl-data/validation/rubik/components/moe_uniform/cllp/4/checkpoint-2372409`          |
|  7 | Rubik   | CLLP        | 5      | `cllp/5/checkpoint-2181045`                   | `./rl-data/validation/rubik/components/moe_uniform/cllp/5/checkpoint-2181045`          |
|  8 | Rubik   | CLLP        | 6      | `cllp/6/checkpoint-2080640`                   | `./rl-data/validation/rubik/components/moe_uniform/cllp/6/checkpoint-2080640`          |
|  9 | Rubik   | CLLP        | 7      | `cllp/7/checkpoint-2062690`                   | `./rl-data/validation/rubik/components/moe_uniform/cllp/7/checkpoint-2062690`          |
| 10 | Rubik   | CLLP        | 8      | `cllp/8/checkpoint-1904720`                   | `./rl-data/validation/rubik/components/moe_uniform/cllp/8/checkpoint-1904720`          |
| 11 | Rubik   | Generator   | 1      | `generator/1/checkpoint-217497`               | `./rl-data/validation/rubik/components/moe_uniform/generator/1/checkpoint-217497`      |
| 12 | Rubik   | Generator   | 2      | `generator/2/checkpoint-217497`               | `./rl-data/validation/rubik/components/moe_uniform/generator/2/checkpoint-217497`      |
| 13 | Rubik   | Generator   | 3      | `generator/3/checkpoint-217497`               | `./rl-data/validation/rubik/components/moe_uniform/generator/3/checkpoint-217497`      |
| 14 | Rubik   | Generator   | 4      | `generator/4/checkpoint-217497`               | `./rl-data/validation/rubik/components/moe_uniform/generator/4/checkpoint-217497`      |
| 15 | Rubik   | Generator   | 5      | `generator/5/checkpoint-310710`               | `./rl-data/validation/rubik/components/moe_uniform/generator/5/checkpoint-310710`      |
| 16 | Rubik   | Generator   | 6      | `generator/6/checkpoint-279639`               | `./rl-data/validation/rubik/components/moe_uniform/generator/6/checkpoint-279639`      |
| 17 | Rubik   | Generator   | 7      | `generator/7/checkpoint-279639`               | `./rl-data/validation/rubik/components/moe_uniform/generator/7/checkpoint-279639`      |
| 18 | Rubik   | Generator   | 8      | `generator/8/checkpoint-279639`               | `./rl-data/validation/rubik/components/moe_uniform/generator/8/checkpoint-279639`      |
| 19 | Rubik   | Policy      | N/A    | `policy/checkpoint-763128`                    | `./rl-data/validation/rubik/components/moe_uniform/policy/checkpoint-763128`           |
| 20 | Rubik   | Value       | N/A    | `value/checkpoint-14504490`                   | `./rl-data/validation/rubik/components/moe_uniform/value/checkpoint-14504490`          |
| 21 | Sokoban | CLLP        | 1      | `cllp/1/checkpoint-149248`                    | `./rl-data/validation/sokoban/components/full_data/cllp/1/checkpoint-149248`           |
| 22 | Sokoban | CLLP        | 16     | `cllp/16/checkpoint-37224`                    | `./rl-data/validation/sokoban/components/full_data/cllp/16/checkpoint-37224`           |
| 23 | Sokoban | CLLP        | 32     | `cllp/32/checkpoint-20045`                    | `./rl-data/validation/sokoban/components/full_data/cllp/32/checkpoint-20045`           |
| 24 | Sokoban | CLLP        | 4      | `cllp/4/checkpoint-587940`                    | `./rl-data/validation/sokoban/components/full_data/cllp/4/checkpoint-587940`           |
| 25 | Sokoban | CLLP        | 8      | `cllp/8/checkpoint-167585`                    | `./rl-data/validation/sokoban/components/full_data/cllp/8/checkpoint-167585`           |
| 26 | Sokoban | Generator   | 1      | `generator/border/1/checkpoint-151712`        | `./rl-data/validation/sokoban/components/full_data/generator/border/1/checkpoint-151712`|
| 27 | Sokoban | Generator   | 16     | `generator/border/16/checkpoint-52150`        | `./rl-data/validation/sokoban/components/full_data/generator/border/16/checkpoint-52150`|
| 28 | Sokoban | Generator   | 32     | `generator/border/32/checkpoint-31290`        | `./rl-data/validation/sokoban/components/full_data/generator/border/32/checkpoint-31290`|
| 29 | Sokoban | Generator   | 4      | `generator/border/4/checkpoint-75856`         | `./rl-data/validation/sokoban/components/full_data/generator/border/4/checkpoint-75856`|
| 30 | Sokoban | Generator   | 8      | `generator/border/8/checkpoint-75856`         | `./rl-data/validation/sokoban/components/full_data/generator/border/8/checkpoint-75856`|
| 31 | Sokoban | Generator   | 1      | `generator/no_border/1/checkpoint-284460`     | `./rl-data/validation/sokoban/components/full_data/generator/no_border/1/checkpoint-284460`|
| 32 | Sokoban | Generator   | 4      | `generator/no_border/4/checkpoint-360316`     | `./rl-data/validation/sokoban/components/full_data/generator/no_border/4/checkpoint-360316`|
| 33 | Sokoban | Generator   | 8      | `generator/no_border/8/checkpoint-284460`     | `./rl-data/validation/sokoban/components/full_data/generator/no_border/8/checkpoint-284460`|
| 34 | Sokoban | Policy      | N/A    | `policy/checkpoint-94820`                     | `./rl-data/validation/sokoban/components/full_data/policy/checkpoint-94820`            |
| 35 | Sokoban | Value       | N/A    | `value/checkpoint-1343100`                    | `./rl-data/validation/sokoban/components/full_data/value/checkpoint-1343100`           |

Made on version v1.0.0 of the CaRL library to ensure reproducibility of algorithms on selected environments, using the components listed above.

**NPuzzle, Sokoban:** [notebooks with evaluation](notebooks/eval.ipynb)

For INT environment, please see [INT repository](https://github.com/mtyrolski/INT).

