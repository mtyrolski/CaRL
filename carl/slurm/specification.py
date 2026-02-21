import os
from dataclasses import dataclass
from typing import get_type_hints

import yaml


@dataclass
class ClusterSpec:
    host: str
    storage_dir: str
    config_dir: str
    repo_url: str
    apptainer_container: str
    apptainer_exec_args: str
    neptune_api_token: str | None
    node_specs: dict
    data_dir: str

    @property
    def repo_name(self) -> str:
        return self.repo_url.split('/')[-1].split('.')[0]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ClusterSpec':
        # check file exists
        if not os.path.exists(yaml_path):
            raise ValueError(f'YAML file {yaml_path} does not exist')

        # load yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f)

        # check for required keys
        required_keys = [
            'host',
            'storage_dir',
            'data_dir',
            'apptainer_container',
            'apptainer_exec_args',
        ]

        if not all(key in yaml_dict for key in required_keys):
            raise ValueError(f'YAML file {yaml_path} must contain keys {required_keys}')

        # load .env file to fullfill secrets
        env_path = '.tokens.env'
        if not os.path.exists(env_path):
            raise ValueError(f'env file {env_path} does not exist')

        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=True)

        # Extend yaml_dict with env variables
        yaml_dict['neptune_api_token'] = os.getenv('NEPTUNE_API_TOKEN')

        return cls(**yaml_dict)


@dataclass
class WorkerTypeSpec:
    carl_worker_name: str
    num_workers: int
    node_spec_name: str


@dataclass
class JobSpec:
    nodes: int
    account: str | None
    partition: str
    job_name: str
    time: int
    cpus_per_task: int
    ntasks_per_node: int
    mem_per_cpu: str
    output: str
    error: str
    gres: str | None

    def to_slurm(self) -> str:
        type_hints = get_type_hints(JobSpec)
        field_names = list(type_hints.keys())

        return '\n'.join([
            f"#SBATCH --{k.replace('_', '-')}={getattr(self, k)}" for k in field_names if getattr(self, k) is not None
        ])

    @classmethod
    def from_specs(cls, cluster_spec: ClusterSpec, worker_specs: list[WorkerTypeSpec]) -> list['JobSpec']:
        job_specs = []

        for ws in worker_specs:
            nodes_count = ws.num_workers
            partitions = cluster_spec.node_specs[ws.node_spec_name]['partition']

            job_name = 'test'
            output = r'hetjob.r%t.%J.out'
            error = r'hetjob.r%t.%J.err'

            gres_count = cluster_spec.node_specs[ws.node_spec_name]['gpus-per-task']
            gres = f'gpu:{gres_count}' if gres_count > 0 else None

            cpus_per_task = cluster_spec.node_specs[ws.node_spec_name]['cpus-per-task']
            mem_per_cpu = cluster_spec.node_specs[ws.node_spec_name]['mem-per-cpu']
            time: int = cluster_spec.node_specs[ws.node_spec_name]['time']
            if 'account' in cluster_spec.node_specs[ws.node_spec_name]:
                account: str | None = cluster_spec.node_specs[ws.node_spec_name]['account']
            else:
                account = None

            job_specs.append(
                cls(
                    nodes=nodes_count,
                    account=account,
                    partition=partitions,
                    job_name=job_name,
                    time=time,
                    mem_per_cpu=mem_per_cpu,
                    ntasks_per_node=1,
                    cpus_per_task=cpus_per_task,
                    output=output,
                    error=error,
                    gres=gres,
                ))
        return job_specs
