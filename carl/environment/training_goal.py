from enum import Enum


class TrainingGoal(Enum):
    POLICY = 'policy'
    VALUE = 'value'
    CLLP = 'cllp'
    GENERATOR = 'generator'
    POLICY_GENERATION = 'policy_generation'
    VALUE_GENERATION = 'value_generation'
    STATE_ACTION_STATE = 'env_simulation'
    STATE_STATE_ACTION = 'return_state_action'
    STATE_ACTION_STATE_GENERATOR = 'return_action_state'
