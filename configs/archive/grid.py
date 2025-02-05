import os

base_cfg: str = """  - algorithm.cllp_path: ["{cllp_path}"]
    algorithm.subgoal_generator_path: ["{generator_path}"]
    algorithm.value_function_path: ["{value_path}"]
    algorithm.planner_class.k: [{k}]
    algorithm.subgoal_generation_kwargs.num_return_sequences: [4]
    algorithm.custom_logger.name: ['{label}']
    algorithm.custom_logger.description: ['{label}']
    algorithm.custom_logger.tags: ['{label}']
"""

value = {
    'super_mini': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/value/50/checkpoint-4580',
    'minimalny': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/value/100/checkpoint-9983',
    'raczej_mini': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/value/200/checkpoint-5499',
    'maly': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/value/1000/checkpoint-47047',
    'pelny': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/value/10000/checkpoint-232061',
}

cllp = {
    'super_mini': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/50/1/checkpoint-5938',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/50/2/checkpoint-154',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/50/4/checkpoint-24',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/50/8/checkpoint-7',
    },
    'minimalny': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/100/1/checkpoint-19968',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/100/2/checkpoint-780',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/100/4/checkpoint-200',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/100/8/checkpoint-30',
    },
    'raczej_mini': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/200/1/checkpoint-48135',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/200/2/checkpoint-760',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/200/4/checkpoint-736',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/200/8/checkpoint-630',
    },
    'maly': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/1000/1/checkpoint-139986',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/1000/2/checkpoint-10148',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/1000/4/checkpoint-5478',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/1000/8/checkpoint-3816',
    },
    'pelny': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/10000/1/checkpoint-123966',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/10000/2/checkpoint-126300',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/10000/4/checkpoint-85696',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/cllp/10000/8/checkpoint-47340',
    },
}

generator = {
    'super_mini': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/50/1/checkpoint-9740',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/50/2/checkpoint-506',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/50/4/checkpoint-542',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/50/8/checkpoint-440',
    },
    'minimalny': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/100/1/checkpoint-2544',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/100/2/checkpoint-2892',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/100/4/checkpoint-2848',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/100/8/checkpoint-11676',
    },
    'raczej_mini': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/200/1/checkpoint-7191',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/200/2/checkpoint-5346',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/200/4/checkpoint-2844',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/200/8/checkpoint-2088',
    },
    'maly': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/1000/1/checkpoint-16426',
        2: None,
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/1000/4/checkpoint-13158',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/1000/8/checkpoint-5031',
    },
    'pelny': {
        1: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/10000/1/checkpoint-43452',
        2: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/10000/2/checkpoint-23004',
        4: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/10000/4/checkpoint-26412',
        8: '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/generator/10000/8/checkpoint-21300',
    },
}

baseline_policy = {
    'super_mini': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/policy/50/checkpoint-32',
    'minimalny': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/policy/100/checkpoint-240',
    'raczej_mini': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/policy/200/checkpoint-510',
    'maly': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/policy/1000/checkpoint-2494',
    'pelny': '/net/pr2/projects/plgrid/plggrl_algo/CaRL/final_components/sokoban/policy/10000/checkpoint-16169',
}

for size, path in value.items():
    print(f'rsync -vuar aresal:{path} final_components/sokoban/value/{size}')
    os.makedirs(f'final_components/sokoban/value/{size}', exist_ok=True)
    os.system(f'rsync -vuar aresal:{path} final_components/sokoban/value/{size}')

for size, path in baseline_policy.items():
    print(f'rsync -vuar aresal:{path} final_components/sokoban/baseline_policy/{size}')
    os.makedirs(f'final_components/sokoban/baseline_policy/{size}', exist_ok=True)
    os.system(f'rsync -vuar aresal:{path} final_components/sokoban/baseline_policy/{size}')

for size, paths in cllp.items():
    for k, path in paths.items():
        print(f'rsync -vuar aresal:{path} final_components/sokoban/cllp/{size}/{k}')
        os.makedirs(f'final_components/sokoban/cllp/{size}/{k}', exist_ok=True)
        os.system(f'rsync -vuar aresal:{path} final_components/sokoban/cllp/{size}/{k}')

for size, paths in generator.items():
    for k, path in paths.items():
        if path is None:
            continue
        print(f'rsync -vuar aresal:{path} final_components/sokoban/generator/{size}/{k}')
        os.makedirs(f'final_components/sokoban/generator/{size}/{k}', exist_ok=True)
        os.system(f'rsync -vuar aresal:{path} final_components/sokoban/generator/{size}/{k}')

# for k in [1, 2, 4, 8]:
#     for ds in ['super_mini', 'minimalny', 'raczej_mini', 'maly', 'pelny']:

#         for model in [generator_path, cllp_path, value_path]:
#             if model is None:

#             os.system(

#         # if generator_path is None or cllp_path is None or value_path is None:
#         #         k=k)
