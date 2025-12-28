# pylint: disable=redefined-outer-name
import argparse
import os
import sys
import tempfile

import click
import git
import numpy as np
import yaml
from loguru import logger
from omegaconf import OmegaConf
from wonderwords import RandomWord

from carl.slurm.connection import Connection
from carl.slurm.grid_search import CarlGrid
from carl.slurm.specification import ClusterSpec
from carl.slurm.specification import JobSpec
from carl.slurm.specification import WorkerTypeSpec


class SlurmLauncher:
    def __init__(
        self,
        cluster_spec: ClusterSpec,
        config_name: str,
        workers_specs: list[WorkerTypeSpec],
        job_specs: list[JobSpec],
        ignore_dirty: bool,
    ):
        self.cluster_spec = cluster_spec
        self.workers_specs = workers_specs
        self.job_specs = job_specs
        self.config_name = config_name
        self.job_name = SlurmLauncher.random_job_name()
        self.experiment_root = os.path.join(self.cluster_spec.storage_dir, self.job_name)
        self.ignore_dirty = ignore_dirty

    @classmethod
    def random_job_name(cls) -> str:
        random_word = RandomWord()
        adjective = random_word.word(include_parts_of_speech=['adjectives'])
        noun = random_word.word(include_parts_of_speech=['nouns'])
        commit_hash = git.Repo('.').head.object.hexsha
        return f'{adjective}_{noun}_{commit_hash[:4]}'

    def _handle_nodes_exit(self) -> str:
        return R"""declare -A process_statuses

for pid in $pids; do
    wait $pid
    status=$?
    if [ $status -ne 0 ]; then
        echo "Process $pid failed with status $status"
        exit $status
    fi
done

exit 0"""

    def _handle_experiment_structure(self):
        data_dir_folder = os.path.basename(self.cluster_spec.data_dir)
        return """
mkdir -p {experiment_root}/$CARL_SLURM_ARRAY_TASK_ID
cp -r {experiment_root}/{repo_name}/* {experiment_root}/$CARL_SLURM_ARRAY_TASK_ID
cp {experiment_root}/{repo_name}/.tokens.env {experiment_root}/$CARL_SLURM_ARRAY_TASK_ID


cp {experiment_root}/configs/$CARL_SLURM_ARRAY_TASK_ID/*.yaml {experiment_root}/$CARL_SLURM_ARRAY_TASK_ID

cd {experiment_root}/$CARL_SLURM_ARRAY_TASK_ID
ln -sf {data_dir} ./{data_dir_folder}

NEPTUNE_CUSTOM_RUN_ID=`date +"%Y%m%d%H%M%s%N" | md5sum`

echo "" >> .tokens.env
echo "NEPTUNE_CUSTOM_RUN_ID=$NEPTUNE_CUSTOM_RUN_ID" >> .tokens.env

pids=\"\" """.format(
            experiment_root=self.experiment_root,
            repo_name=self.cluster_spec.repo_name,
            data_dir=self.cluster_spec.data_dir,
            data_dir_folder=data_dir_folder,
        )

    def _handle_node_task(self, worker_number: int, het_group: int, worker_type_spec: WorkerTypeSpec, all_workers: int):
        job_options = ' '.join([
            f'--{k}={v}' for k, v in self.cluster_spec.node_specs[worker_type_spec.node_spec_name].items()
            if k != 'gpus-per-task'
        ])
        additional_args = ' '.join(self.cluster_spec.apptainer_exec_args)

        node_name = f'${{nodes_het_group_{het_group}[{worker_number}]}}'

        het_group_srun_flag = f'--het-group={het_group}' if len(self.workers_specs) > 1 else ''

        carl_env_vars = ' '.join([
            f'CARL_ALL_NODES_COUNT={all_workers}',
            f'CARL_HET_GROUP_ID={het_group}',
            f'CARL_LOCAL_WORKER_ID={worker_number}',
            f'CARL_N_NODES_IN_GROUP={worker_type_spec.num_workers}',
        ])

        srun_command = (
            "NEPTUNE_API_TOKEN={neptune_api_token} srun {het_group_srun_flag} -w {node_name} {job_options} "
            "singularity exec {additional_args} --pwd {experiment_root}/$CARL_SLURM_ARRAY_TASK_ID --nv "
            "\"{singularity_image_path}\" /bin/bash -c "
            "\"{carl_env_vars} PYTHONPATH=. python3 -m carl.run --config-dir {config_dir} --config-name {config_name}\" &"
        )

        return "\n".join(
            [
                "echo \"Starting {worker_number} worker from het group {het_group} on node {node_name}\"",
                srun_command,
                "pids=\"$pids $!\"",
                "",
            ]
        ).format(
            config_dir='.',
            config_name=worker_type_spec.carl_worker_name,
            worker_number=worker_number,
            job_options=job_options,
            additional_args=additional_args,
            experiment_root=self.experiment_root,
            singularity_image_path=self.cluster_spec.apptainer_container,
            neptune_api_token=self.cluster_spec.neptune_api_token,
            node_name=node_name,
            het_group=het_group,
            het_group_srun_flag=het_group_srun_flag,
            carl_env_vars=carl_env_vars,
        )

    def _handle_nodes_tasks(self):
        node_tasks = []

        all_workers = sum([worker_type_spec.num_workers for worker_type_spec in self.workers_specs])

        for het_group, worker_type_spec in enumerate(self.workers_specs):
            for node_idx in range(worker_type_spec.num_workers):
                node_tasks.append(self._handle_node_task(node_idx, het_group, worker_type_spec, all_workers))

        return '\n\n'.join(node_tasks)

    def _generate_sbatch(self, sbatch_idx: int):
        intro = '#!/bin/bash'

        het_jobs_config = '\n\n#SBATCH hetjob\n\n'.join([job_specs.to_slurm() for job_specs in self.job_specs])

        carl_variables = '\n'.join([
            f'export CARL_SLURM_ARRAY_TASK_ID={sbatch_idx}',
        ])
        carl_hostnames = '\n'.join([
            (
                f"IFS=' ' read -ra nodes_het_group_{het_group_idx} <<< "
                f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{het_group_idx} | "
                "tr '\n' ' ' | sed 's/ $//')"
            )
            for het_group_idx in range(len(self.workers_specs))
        ])

        carl_print_het_groups = '\n'.join([
            f'echo Nodes from {het_group_idx}: $SLURM_JOB_NODELIST_HET_GROUP_{het_group_idx}'
            for het_group_idx in range(len(self.workers_specs))
        ])

        carl_print_hostname = '\n'.join([
            f'echo "Het group {het_group_idx} has nodes $nodes_het_group_{het_group_idx}"'
            for het_group_idx in range(len(self.workers_specs))
        ])

        exp_structure = self._handle_experiment_structure()
        nodes_tasks = self._handle_nodes_tasks()
        nodes_exit = self._handle_nodes_exit()

        # wrap it up
        components = [
            intro,
            het_jobs_config,
            carl_variables,
            carl_hostnames,
            carl_print_het_groups,
            carl_print_hostname,
            exp_structure,
            nodes_tasks,
            nodes_exit,
        ]

        return '\n\n'.join(components)

    def launch(self, n_sample: int, dry_run: bool = False, only_first: bool = False):
        file = f'{self.config_name}.yaml'
        grid_search = CarlGrid.from_file(file)

        grids = list(grid_search.iter_grid())

        if n_sample > 0:
            logger.info(f'Sampling {n_sample} experiments from {len(grids)}')
            idxs = np.random.choice(len(grids), n_sample, replace=False)
            grids = [grids[idx] for idx in idxs]

        if only_first:
            if len(grids) > 1:
                logger.warning(
                    f'Only first experiment will be run. Running only first of {len(grids)} **sampled** grids')

            logger.info(f'Only first experiment will be run. Reducing from {len(grids)} to 1')
            grids = grids[:1]

        experiment_count = len(grids)

        logger.info(f'Experiment count: {experiment_count}')

        if experiment_count == 0:
            logger.error('No experiments to run')
            logger.error('Add carl_grid to your config file')
            logger.warning("If you don't want to run grid search (fix after deadline), put dummy like this:\ncarl_grid:\
  algorithm.n_solving_cores: [40]\
")
            return

        # TODO: config version for each worker.
        logger.info('Generating sbatch file...')

        Connection.exec_on_rem_workspace(
            self.cluster_spec.host,
            self.cluster_spec.storage_dir,
            ['echo "Connection to cluster established"'],
        )

        if (not self.ignore_dirty) and git.Repo('.').is_dirty():
            if click.confirm(
                    'Local repository is dirty. Cluster will pull last pushed changes. Do you want to continue?',
                    default=True,
            ):
                pass
            else:
                print('Aborting...')
                sys.exit(1)
        Connection.exec_on_rem_workspace(
            self.cluster_spec.host,
            self.cluster_spec.storage_dir,
            [f'mkdir -p {self.job_name}'],
        )

        current_branch = git.Repo('.').active_branch.name
        git_repo_hash = git.Repo('.').head.object.hexsha

        # Pull remotely repository
        experiment_root = os.path.join(self.cluster_spec.storage_dir, self.job_name)
        Connection.exec_on_rem_workspace(
            self.cluster_spec.host,
            experiment_root,
            [f'git clone {self.cluster_spec.repo_url}'],
        )

        main_repo_dir = os.path.join(experiment_root, self.cluster_spec.repo_name)
        Connection.exec_on_rem_workspace(
            self.cluster_spec.host,
            main_repo_dir,
            [f'git checkout {current_branch}'],
        )
        Connection.exec_on_rem_workspace(
            self.cluster_spec.host,
            main_repo_dir,
            [f'git reset --hard {git_repo_hash}'],
        )

        # Send .tokens.env to remote repo
        Connection.send_to_server('.tokens.env', self.cluster_spec.host, main_repo_dir)

        sbatch_file_name: str | None = None
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # Create a sbatch file
            for sbatch_idx in range(experiment_count):
                sbatch_content = self._generate_sbatch(sbatch_idx)
                if sbatch_idx == 0:
                    logger.info(f'Sbatch[0] file generated:\n{sbatch_content}')

                    if experiment_count > 1:
                        logger.warning(
                            f'Omitting printing of other sbatch files ({experiment_count-1}/{experiment_count})')

                sbatch_file_name = f'sbatch{sbatch_idx}.sh'
                sbatch_file_path = os.path.join(tmp_dir_name, sbatch_file_name)
                with open(sbatch_file_path, 'w+') as f:
                    f.write(sbatch_content)

            # Grid create configs
            logger.info('Creating configs...')
            tmp_config_dir = os.path.join(tmp_dir_name, 'configs')
            # Create config dir
            os.makedirs(tmp_config_dir, exist_ok=True)
            for task_array_idx, workername2config_dict in enumerate(grids):
                os.makedirs(os.path.join(tmp_config_dir, str(task_array_idx)), exist_ok=True)
                for carl_worker_name, experiment_config in workername2config_dict.items():
                    config_file_path = os.path.join(tmp_config_dir, str(task_array_idx), f'{carl_worker_name}.yaml')
                    carl_config_dict = OmegaConf.to_container(experiment_config, resolve=True)
                    with open(config_file_path, 'w+') as f:
                        yaml.dump(carl_config_dict, f)

            Connection.send_to_server(
                os.path.join(tmp_dir_name, 'configs'),
                self.cluster_spec.host,
                experiment_root,
                recursive=True,
            )

            # Send sbatch file to cluster
            logger.info(f'Sending sbatch files ({experiment_count}) to cluster')
            os.system(f'rsync -uvar {tmp_dir_name}/sbatch*.sh {self.cluster_spec.host}:{experiment_root}')

            # print tree of temp dir
            logger.info(f'Tree of temp dir {tmp_dir_name}')
            os.system(f'tree {tmp_dir_name}')
            os.system(f'cp -r {tmp_dir_name} ./{self.job_name}')

        logger.info(f'Working directory: {experiment_root}')
        if sbatch_file_name is not None:
            logger.info(f'Launching sbatch file {sbatch_file_name} on cluster')

        # Call a sbatch
        if not dry_run:
            Connection.exec_on_rem_workspace(
                self.cluster_spec.host,
                experiment_root,
                [f'sbatch sbatch{sbatch_idx}.sh' for sbatch_idx in range(experiment_count)],
            )


def algo_count_nodespec(arg):
    args = arg.split(';')
    if len(args) != 3:
        raise argparse.ArgumentTypeError(
            'Algo nodespec must be of the form <config_name>;<num_workers>;<node_spec_name>')

    carl_worker_name, num_workers, node_spec_name = args
    num_workers = int(num_workers)
    return WorkerTypeSpec(carl_worker_name, num_workers, node_spec_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-config', type=str, required=True)
    parser.add_argument('--job-config', type=str, required=True)    # path to config
    parser.add_argument('--worker', type=algo_count_nodespec, action='append',
                        required=True)    # worker_name, num_workers, node_spec_name(cluster)
    parser.add_argument('--ignore-dirty', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only-first', action='store_true')
    parser.add_argument('--n-sample', type=int, default=-1)
    args = parser.parse_args()

    # set loguru logger lvl to info
    logger.remove()
    logger.add(sys.stdout, level='INFO')

    cluster_spec = ClusterSpec.from_yaml(args.cluster_config)
    job_specs: list[JobSpec] = JobSpec.from_specs(cluster_spec, args.worker)

    if len(args.worker) == 0:
        raise ValueError('At least one worker must be specified')

    worker_type_specs = args.worker
    config_name = args.job_config

    SlurmLauncher(cluster_spec, config_name, worker_type_specs, job_specs,
                  args.ignore_dirty).launch(args.n_sample, args.dry_run, args.only_first)
