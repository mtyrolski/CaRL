# Embedding-Based Hierarchical Search in CaRL using HuggingFace Transformers

This document describes the new embedding-based hierarchical search capabilities added to CaRL, which enable hierarchical planning in continuous latent space using HuggingFace transformer backbones rather than discrete state space.

## Overview

Traditional hierarchical reinforcement learning algorithms like AdaSubS operate in discrete state spaces, which limits their applicability to continuous environments and can be computationally expensive for complex state representations. The embedding-based extensions in CaRL address these limitations by:

1. **Learning compressed state representations** using HuggingFace BERT-based Autoencoders (AE) and Variational Autoencoders (VAE)
2. **Operating in embedding space** for subgoal generation and navigation using BART and BERT architectures
3. **Enabling continuous latent space search** for hierarchical planning with transformer backbones
4. **Supporting real-time RL** through efficient HuggingFace model operations

## Architecture

The embedding-based hierarchical search consists of three main components using HuggingFace transformers:

### 1. State Embedding Models

**Location**: `carl/components/state_embeddings.py`

- **HFStateAutoencoder**: BERT-based autoencoder for deterministic state embeddings
- **HFStateVAE**: BERT-based variational autoencoder with regularized probabilistic embeddings  

**Key Features**:
- Built on HuggingFace BERT architecture for robust transformer-based encoding
- Configurable through `StateEmbeddingConfig` following HuggingFace patterns
- Support for `from_pretrained()` loading and saving
- Built-in reconstruction losses for training

### 2. Embedding Generators

**Location**: `carl/components/embedding_generator.py`

Generate subgoal embeddings from current state embeddings using BART:

- **HFEmbeddingGenerator**: BART-based generator with embedding projection layers
- **HFTransformerEmbeddingGenerator**: BART encoder-decoder for embedding-to-embedding generation
- **HFAutoregressiveEmbeddingGenerator**: BART-based autoregressive generator for multi-step planning

**Key Features**:
- Built on HuggingFace BART architecture for sequence-to-sequence generation
- Multiple k-step values for different planning horizons
- Attention mechanisms built into BART backbone
- Configurable through `EmbeddingGeneratorConfig`

### 3. Embedding-Conditioned CLLPs

**Location**: `carl/components/embedding_cllp.py`

Conditional Low-Level Policies that operate on embeddings using BERT:

- **HFEmbeddingConditionedCLLP**: BERT-based CLLP with cross-attention between state and subgoal embeddings
- **HFHierarchicalEmbeddingCLLP**: Multi-scale BERT CLLP for hierarchical subgoals
- **HFProgressAwareCLLP**: BERT CLLP with progress tracking towards subgoals

**Key Features**:
- Built on HuggingFace BERT architecture for classification tasks
- Cross-attention mechanisms for state-subgoal conditioning
- Configurable through `EmbeddingCLLPConfig`

## Training Pipeline

### 1. State Embedding Training

Train HuggingFace BERT-based autoencoders to learn compressed state representations:

```bash
# Train BERT-based autoencoder
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_state_embedding_ae

# Train BERT-based variational autoencoder  
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_state_embedding_vae
```

**Training Goals**:
- `STATE_EMBEDDING_AE`: BERT autoencoder reconstruction
- `STATE_EMBEDDING_VAE`: BERT VAE reconstruction + KL regularization

### 2. Embedding Generator Training

Train BART-based generators to predict subgoal embeddings:

```bash
# Train BART generators for different k values
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_generator_k4
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_generator_k8
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_generator_k16
```

**Training Goal**: `EMBEDDING_GENERATOR`
**Architecture**: BART encoder-decoder with embedding projection layers

### 3. Embedding CLLP Training

Train BERT-based CLLPs to navigate in embedding space:

```bash
# Train BERT-based embedding-conditioned CLLP
python -m carl.run --config-dir configs/offline_training/sokoban --config-name sokoban_train_embedding_cllp
```

**Training Goal**: `EMBEDDING_CLLP`
**Architecture**: BERT for sequence classification with embedding conditioning

## Inference Components

### HuggingFace Embedding Subgoal Generators

**Location**: `carl/inference_components/embedding_subgoal_generator.py`

- **HFEmbeddingSubgoalGenerator**: Integrates HuggingFace models with CaRL's inference framework
- **AdaptiveHFEmbeddingSubgoalGenerator**: Multi-k value generator using different BART models
- **HybridHFSubgoalGenerator**: Combines discrete and HuggingFace embedding-based generation

### HuggingFace Embedding CLLPs

**Location**: `carl/inference_components/embedding_conditional_low_level_policy.py`

- **HFEmbeddingConditionalLowLevelPolicy**: Base HuggingFace embedding CLLP for inference
- **ProgressAwareHFEmbeddingCLLP**: BERT CLLP with progress monitoring
- **MultiScaleHFEmbeddingCLLP**: Multi-scale planning using BERT in embedding space
- **HFEmbeddingCLLPValidator**: Validates subgoal reachability using HuggingFace models

## Configuration

### Model Configuration

Example BERT autoencoder configuration:
```yaml
model:
  _partial_: True
  _target_: carl.components.state_embeddings.HFStateAutoencoder
config:
  _target_: carl.components.state_embeddings.StateEmbeddingConfig
  vocab_size: 144  # 12x12 Sokoban board
  hidden_size: 512
  num_hidden_layers: 6
  num_attention_heads: 8
  intermediate_size: 2048
  latent_dim: 64
```

Example BART embedding generator configuration:
```yaml
model:
  _partial_: True
  _target_: carl.components.embedding_generator.HFEmbeddingGenerator
config:
  _target_: carl.components.embedding_generator.EmbeddingGeneratorConfig
  vocab_size: 144
  d_model: 512
  encoder_layers: 4
  decoder_layers: 4
  encoder_attention_heads: 8
  decoder_attention_heads: 8
  embedding_dim: 64
```

### Training Configuration

All embedding-based training uses HuggingFace infrastructure:

- **Models**: `transformers.Trainer` with HuggingFace `PreTrainedModel` subclasses
- **Metrics**: Custom metrics for reconstruction quality, embedding similarity, action accuracy  
- **Logging**: Neptune integration for experiment tracking
- **Data Loading**: Extended `GameDataModule` with new tokenization methods

## Metrics and Evaluation

### State Embedding Metrics
- **Reconstruction loss**: Cross-entropy loss for BERT autoencoder
- **KL divergence**: For BERT VAE regularization
- **Embedding quality**: Validation on reconstruction tasks

### Generator Metrics  
- **Embedding MSE/MAE**: Accuracy of subgoal embedding prediction using BART
- **Generation quality**: Validation of generated subgoal embeddings
- **Cross-attention analysis**: Attention patterns in BART encoder-decoder

### CLLP Metrics
- **Action accuracy**: BERT classification accuracy for action prediction
- **Cross-attention effectiveness**: Analysis of state-subgoal attention patterns
- **Progress tracking**: Convergence towards subgoals (when supported)

## Advantages

1. **HuggingFace Ecosystem**: Leverages proven transformer architectures (BERT, BART)
2. **Robust Training**: Benefits from HuggingFace's training infrastructure and optimizations
3. **Pretrained Components**: Can leverage pretrained BERT/BART weights for initialization
4. **Scalability**: Transformer architectures scale well to larger problems
5. **Attention Mechanisms**: Built-in attention for long-range dependencies
6. **Model Hub Integration**: Easy saving/loading through HuggingFace model hub

## Usage Examples

### Basic HuggingFace Embedding-Based Search

```python
from carl.inference_components.embedding_subgoal_generator import HFEmbeddingSubgoalGenerator
from carl.inference_components.embedding_conditional_low_level_policy import HFEmbeddingConditionalLowLevelPolicy
from carl.components.state_embeddings import HFStateAutoencoder
from carl.components.embedding_generator import HFEmbeddingGenerator
from carl.components.embedding_cllp import HFEmbeddingConditionedCLLP

# Create HuggingFace embedding-based hierarchical planning pipeline
# Models are loaded using from_pretrained()
subgoal_generator = HFEmbeddingSubgoalGenerator(
    embedding_generator=HFEmbeddingGenerator,
    state_embedding_model=HFStateAutoencoder,
    path_to_generator_weights="path/to/bart/generator/weights",
    path_to_embedding_weights="path/to/bert/embedding/weights",
    env=env
)

embedding_cllp = HFEmbeddingConditionalLowLevelPolicy(
    cllp_network_class=HFEmbeddingConditionedCLLP,
    embedding_model_class=HFStateAutoencoder,
    path_to_cllp_weights="path/to/bert/cllp/weights", 
    path_to_embedding_weights="path/to/bert/embedding/weights",
    env=env
)
```

### Adaptive Multi-K HuggingFace Search

```python
from carl.inference_components.embedding_subgoal_generator import AdaptiveHFEmbeddingSubgoalGenerator
from carl.components.embedding_generator import (
    HFEmbeddingGenerator, 
    HFTransformerEmbeddingGenerator,
    HFAutoregressiveEmbeddingGenerator
)

adaptive_generator = AdaptiveHFEmbeddingSubgoalGenerator(
    generator_k_list=[4, 8, 16],
    paths_to_generator_weights=[
        "path/to/bart/k4/weights",
        "path/to/bart/k8/weights", 
        "path/to/bart/k16/weights"
    ],
    path_to_embedding_weights="path/to/bert/embedding/weights",
    env=env,
    generator_classes=[
        HFEmbeddingGenerator,  # k=4: Basic BART generator
        HFTransformerEmbeddingGenerator,  # k=8: Advanced BART
        HFAutoregressiveEmbeddingGenerator,  # k=16: Autoregressive BART
    ]
)
```

## Future Extensions

1. **Pretrained Initialization**: Use pretrained BERT/BART weights for better initialization
2. **Model Hub Integration**: Save/load models through HuggingFace model hub
3. **Advanced Attention**: Leverage transformer attention for better state-subgoal relationships
4. **Cross-Domain Transfer**: Use pretrained transformers for transfer learning
5. **Larger Models**: Scale to larger BERT/BART variants for improved performance

## References

- AdaSubS: Adaptive Subgoal Search for Hierarchical Reinforcement Learning
- CaRL: Compositional Autoregressive Reinforcement Learning  
- HuggingFace Transformers: BERT and BART architectures
- Attention Is All You Need: Transformer architecture

---

For more information, see the example configurations in `configs/offline_training/sokoban/` and the implementation details in `carl/components/` and `carl/inference_components/`.