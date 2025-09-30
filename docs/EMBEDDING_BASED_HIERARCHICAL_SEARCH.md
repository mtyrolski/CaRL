# Embedding-Based Hierarchical Search in CaRL

This document describes the new embedding-based hierarchical search capabilities added to CaRL, which enable hierarchical planning in continuous latent space rather than discrete state space.

## Overview

Traditional hierarchical reinforcement learning algorithms like AdaSubS operate in discrete state spaces, which limits their applicability to continuous environments and can be computationally expensive for complex state representations. The embedding-based extensions in CaRL address these limitations by:

1. **Learning compressed state representations** using Autoencoders (AE) and Variational Autoencoders (VAE)
2. **Operating in embedding space** for subgoal generation and navigation
3. **Enabling continuous latent space search** for hierarchical planning
4. **Supporting real-time RL** through efficient embedding-based operations

## Architecture

The embedding-based hierarchical search consists of three main components:

### 1. State Embedding Models

**Location**: `carl/components/state_embeddings.py`

- **StateAutoencoder**: Standard autoencoder for deterministic state embeddings
- **StateVAE**: Variational autoencoder with regularized probabilistic embeddings  
- **Conv2DStateAutoencoder**: Convolutional autoencoder for 2D spatial states (e.g., Sokoban boards)

**Key Features**:
- Configurable architecture (hidden dimensions, dropout, etc.)
- Support for both fully connected and convolutional encoders
- Built-in reconstruction losses for training

### 2. Embedding Generators

**Location**: `carl/components/embedding_generator.py`

Generate subgoal embeddings from current state embeddings:

- **EmbeddingGenerator**: Basic generator with optional attention mechanism
- **TransformerEmbeddingGenerator**: Transformer-based generator for sequence modeling
- **AutoregressiveEmbeddingGenerator**: LSTM-based generator for multi-step planning

**Key Features**:
- Multiple k-step values for different planning horizons
- Attention mechanisms for long-range dependencies
- Contrastive and embedding regularization losses

### 3. Embedding-Conditioned CLLPs

**Location**: `carl/components/embedding_cllp.py`

Conditional Low-Level Policies that operate on embeddings:

- **EmbeddingConditionedCLLP**: Basic CLLP with cross-attention between state and subgoal embeddings
- **HierarchicalEmbeddingCLLP**: Multi-scale CLLP for hierarchical subgoals
- **ProgressAwareCLLP**: CLLP with progress tracking towards subgoals
- **ResidualEmbeddingCLLP**: Deep CLLP with residual connections

## Training Pipeline

### 1. State Embedding Training

Train autoencoders to learn compressed state representations:

```bash
# Train standard autoencoder
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_state_embedding_ae

# Train variational autoencoder  
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_state_embedding_vae
```

**Training Goals**:
- `STATE_EMBEDDING_AE`: Autoencoder reconstruction
- `STATE_EMBEDDING_VAE`: VAE reconstruction + KL regularization

### 2. Embedding Generator Training

Train generators to predict subgoal embeddings:

```bash
# Train generators for different k values
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_generator_k4
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_generator_k8
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_generator_k16
```

**Training Goal**: `EMBEDDING_GENERATOR`

### 3. Embedding CLLP Training

Train CLLPs to navigate in embedding space:

```bash
# Train embedding-conditioned CLLP
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_cllp
```

**Training Goal**: `EMBEDDING_CLLP`

## Inference Components

### Embedding Subgoal Generators

**Location**: `carl/inference_components/embedding_subgoal_generator.py`

- **EmbeddingSubgoalGenerator**: Integrates with CaRL's inference framework
- **AdaptiveEmbeddingSubgoalGenerator**: Multi-k value generator
- **HybridSubgoalGenerator**: Combines discrete and embedding-based generation

### Embedding CLLPs

**Location**: `carl/inference_components/embedding_conditional_low_level_policy.py`

- **EmbeddingConditionalLowLevelPolicy**: Base embedding CLLP for inference
- **ProgressAwareEmbeddingCLLP**: CLLP with progress monitoring
- **MultiScaleEmbeddingCLLP**: Multi-scale planning in embedding space
- **EmbeddingCLLPValidator**: Validates subgoal reachability

## Configuration

### Model Configuration

Example autoencoder configuration:
```yaml
model:
  _partial_: True
  _target_: carl.components.state_embeddings.StateAutoencoder
  input_dim: 144  # 12x12 Sokoban board
  embedding_dim: 64
  hidden_dims: [512, 256, 128]
  dropout: 0.1
```

Example embedding generator configuration:
```yaml
model:
  _partial_: True
  _target_: carl.components.embedding_generator.EmbeddingGenerator
  embedding_dim: 64
  hidden_dims: [512, 256]
  k_steps: 4
  dropout: 0.1
  use_attention: True
```

### Training Configuration

All embedding-based training uses the same infrastructure as existing CaRL components:

- **Metrics**: Custom metrics for reconstruction quality, embedding similarity, action accuracy
- **Logging**: Neptune integration for experiment tracking
- **Data Loading**: Extended `GameDataModule` with new tokenization methods

## Metrics and Evaluation

### State Embedding Metrics
- **Reconstruction MSE/MAE**: Quality of state reconstruction
- **R-squared**: Coefficient of determination for reconstruction
- **Embedding norm statistics**: Stability of learned embeddings

### Generator Metrics  
- **Embedding MSE/MAE**: Accuracy of subgoal embedding prediction
- **Cosine similarity**: Semantic similarity between predicted and target embeddings
- **Embedding norm statistics**: Consistency of generated embeddings

### CLLP Metrics
- **Action accuracy**: Success rate of action prediction
- **Per-class accuracy**: Performance breakdown by action type
- **Progress tracking**: Convergence towards subgoals (when supported)

## Advantages

1. **Continuous State Support**: Works with continuous and high-dimensional state spaces
2. **Computational Efficiency**: Operations in compact embedding space
3. **Real-time RL**: Enables online learning and adaptation
4. **Hierarchical Planning**: Multiple levels of abstraction through different k values
5. **Generalization**: Learned embeddings can generalize across similar states
6. **Scalability**: Embedding dimension much smaller than state dimension

## Usage Examples

### Basic Embedding-Based Search

```python
from carl.inference_components.embedding_subgoal_generator import EmbeddingSubgoalGenerator
from carl.inference_components.embedding_conditional_low_level_policy import EmbeddingConditionalLowLevelPolicy
from carl.components.state_embeddings import StateAutoencoder
from carl.components.embedding_generator import EmbeddingGenerator
from carl.components.embedding_cllp import EmbeddingConditionedCLLP

# Initialize components
embedding_model = StateAutoencoder(input_dim=144, embedding_dim=64)
generator_model = EmbeddingGenerator(embedding_dim=64, k_steps=4)
cllp_model = EmbeddingConditionedCLLP(embedding_dim=64, num_actions=4)

# Create inference components
subgoal_generator = EmbeddingSubgoalGenerator(
    embedding_generator=EmbeddingGenerator,
    state_embedding_model=StateAutoencoder,
    path_to_generator_weights="path/to/generator/weights",
    path_to_embedding_weights="path/to/embedding/weights",
    env=env
)

embedding_cllp = EmbeddingConditionalLowLevelPolicy(
    cllp_network_class=EmbeddingConditionedCLLP,
    embedding_model_class=StateAutoencoder,
    path_to_cllp_weights="path/to/cllp/weights", 
    path_to_embedding_weights="path/to/embedding/weights",
    env=env
)
```

### Adaptive Multi-K Search

```python
from carl.inference_components.embedding_subgoal_generator import AdaptiveEmbeddingSubgoalGenerator

adaptive_generator = AdaptiveEmbeddingSubgoalGenerator(
    generator_k_list=[4, 8, 16],
    paths_to_generator_weights=[
        "path/to/k4/weights",
        "path/to/k8/weights", 
        "path/to/k16/weights"
    ],
    path_to_embedding_weights="path/to/embedding/weights",
    env=env
)
```

## Future Extensions

1. **Multi-Modal Embeddings**: Support for different state modalities
2. **Hierarchical Embeddings**: Multiple embedding scales for different abstraction levels
3. **Online Embedding Learning**: Continuous adaptation of embeddings during inference
4. **Cross-Domain Transfer**: Embedding transfer between different environments
5. **Uncertainty Quantification**: Better uncertainty modeling in VAE-based approaches

## References

- AdaSubS: Adaptive Subgoal Search for Hierarchical Reinforcement Learning
- CaRL: Compositional Autoregressive Reinforcement Learning
- Hierarchical Reinforcement Learning with Latent Space Planning

---

For more information, see the example configurations in `configs/offline_training/sokoban/` and the implementation details in `carl/components/` and `carl/inference_components/`.