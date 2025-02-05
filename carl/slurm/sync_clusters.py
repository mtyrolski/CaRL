import argparse
import json
import os

from carl.slurm.connection import Connection

_USERNAME_REMOTE = 'host_name'
_TREE_ROOT = 'tree_root'


def sync_clusters(cfg, target_cluster: str, local_path: str, exclude_cmd: str):
    target_cluster_host = cfg['clusters'][target_cluster][_USERNAME_REMOTE]
    target_tree_root = cfg['clusters'][target_cluster][_TREE_ROOT]

    sync_paths = cfg['sync_paths']
    for sync_path in sync_paths:
        # rsync from local to remote
        Connection.exec_on_rem_workspace(target_cluster_host, target_tree_root, [f'mkdir -p {sync_path}'])
        os.system(
            f'rsync -vuar {os.path.join(local_path, sync_path)}/* {target_cluster_host}:{os.path.join(target_tree_root, sync_path)} {exclude_cmd}'
        )


def main(args):
    config_path = args.config
    source_cluster = args.source_cluster
    target_clusters = args.target_cluster

    with open(config_path, 'r', encoding='utf-8') as cfg_f:
        content = cfg_f.read()
    cfg = json.loads(content)

    src_cluster_host = cfg['clusters'][source_cluster][_USERNAME_REMOTE]
    src_tree_root = cfg['clusters'][source_cluster][_TREE_ROOT]
    local_path = cfg['local'][_TREE_ROOT]
    ignore_paths = cfg['ignore']

    exclude_cmd = ' '.join([f'--exclude *{ignore_path}*' for ignore_path in ignore_paths])

    if not os.path.exists(local_path):
        print("Local path doesn't exists, recreating it")
        os.makedirs(local_path, exist_ok=True)

    # rsync from remote to local
    for sync_path in cfg['sync_paths']:
        os.makedirs(os.path.join(local_path, sync_path), exist_ok=True)
        os.system(
            f'rsync -vuar {src_cluster_host}:{os.path.join(src_tree_root, sync_path)}/* {os.path.join(local_path, sync_path)} {exclude_cmd}'
        )

    for target in target_clusters:
        sync_clusters(cfg_f, target, local_path, exclude_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to clusters config file', required=True, type=str)
    parser.add_argument('-s', '--source-cluster', help='Name of cluster which is synced', required=True, type=str)
    parser.add_argument(
        '-t',
        '--target-cluster',
        action='append',
        help='Names of clusters to be synced',
        required=True,
    )
    _args = parser.parse_args()
    main(_args)
