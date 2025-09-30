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
    # New training goals for embedding-based models
    STATE_EMBEDDING_AE = 'state_embedding_ae'  # Standard autoencoder for state embeddings
    STATE_EMBEDDING_VAE = 'state_embedding_vae'  # Variational autoencoder for state embeddings
    EMBEDDING_GENERATOR = 'embedding_generator'  # Generator operating in embedding space
    EMBEDDING_CLLP = 'embedding_cllp'  # CLLP conditioned on embeddings
