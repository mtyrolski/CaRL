import os

CARL_HET_GROUP_ID = 'CARL_HET_GROUP_ID'
CARL_WORKER_LOCAL_ID = 'CARL_LOCAL_WORKER_ID'
CARL_N_NODES_IN_GROUP = 'CARL_N_NODES_IN_GROUP'
CARL_ALL_NODES_COUNT = 'CARL_ALL_NODES_COUNT'


def get_current_hetgroup_id() -> int:
    return int(os.environ.get(CARL_HET_GROUP_ID, '0'))


def get_current_node_within_hetgroup() -> int:
    return int(os.environ.get(CARL_WORKER_LOCAL_ID, '0'))


def get_n_nodes_in_hetgroup() -> int:
    return int(os.environ.get(CARL_N_NODES_IN_GROUP, '1'))


def get_all_nodes_count() -> int:
    return int(os.environ.get(CARL_ALL_NODES_COUNT, '1'))
