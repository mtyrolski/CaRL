"""Embedding-based subgoal generator for CaRL inference framework.

This module provides subgoal generators that operate in learned embedding
space, integrating with CaRL's existing inference component architecture.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.inference_components.component import InferenceComponent
from carl.inference_components.subgoal_generator import SubgoalGenerator
from carl.solver.nodes import GeneratedSubgoal, SearchTreeNode
from carl.components.state_embeddings import BaseStateEmbedding
from carl.components.embedding_generator import EmbeddingGenerator


class EmbeddingSubgoalGenerator(SubgoalGenerator):
    """Subgoal generator that operates in embedding space.
    
    Uses a pretrained state embedding model (AE/VAE) and an embedding
    generator to produce subgoals in latent space, then maps them back
    to state space for use in search.
    """
    
    def __init__(
        self,
        embedding_generator: type[nn.Module],
        state_embedding_model: type[nn.Module],
        path_to_generator_weights: str,
        path_to_embedding_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__(
            embedding_generator,
            path_to_generator_weights,
            env,
            subgoal_generation_kwargs
        )
        
        self.state_embedding_model = state_embedding_model
        self.path_to_embedding_weights = path_to_embedding_weights
        self.embedding_model: Optional[BaseStateEmbedding] = None
        self.embedding_generator_model: Optional[EmbeddingGenerator] = None
        
        # Default generation parameters
        self.default_k_steps = subgoal_generation_kwargs.get('k_steps', 4) if subgoal_generation_kwargs else 4
        self.num_subgoals = subgoal_generation_kwargs.get('num_subgoals', 5) if subgoal_generation_kwargs else 5
        
    def construct_network(self) -> None:
        """Construct both embedding and generator networks."""
        # Load state embedding model
        self.embedding_model = self.instantiate_network(
            self.state_embedding_model,
            self.path_to_embedding_weights
        )
        
        # Load embedding generator model
        self.embedding_generator_model = self.instantiate_network(
            self.generator,
            self.path_to_generator_weights
        )
        
    def get_network(self) -> dict[str, nn.Module]:
        """Return both networks."""
        assert self.embedding_model is not None, "Embedding model not constructed"
        assert self.embedding_generator_model is not None, "Generator model not constructed"
        
        return {
            'embedding_model': self.embedding_model,
            'generator_model': self.embedding_generator_model
        }
        
    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """Generate subgoals in embedding space and convert back to state space.
        
        Args:
            node: Current search tree node
            
        Returns:
            List of generated subgoals in state space
        """
        assert self.embedding_model is not None, "Embedding model not constructed"
        assert self.embedding_generator_model is not None, "Generator model not constructed"
        
        current_state = node.state
        
        # Convert current state to tensor if needed
        if not isinstance(current_state, torch.Tensor):
            # Use tokenizer to convert state to tensor
            state_tensor, _ = self.env.tokenizer.x_y_tokenizer(
                current_state, current_state, 'state_embedding_ae'
            )
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        else:
            state_tensor = current_state.unsqueeze(0) if current_state.dim() == 1 else current_state
            
        # Encode current state to embedding
        with torch.no_grad():
            current_embedding = self.embedding_model.encode(state_tensor)
            
            # Generate multiple subgoal embeddings
            subgoal_embeddings = []
            for _ in range(self.num_subgoals):
                # Sample different k values or add noise for diversity
                k_steps = torch.tensor([self.default_k_steps], device=current_embedding.device)
                subgoal_emb = self.embedding_generator_model(current_embedding, k_steps)
                subgoal_embeddings.append(subgoal_emb)
            
            # Decode embeddings back to state space
            subgoals = []
            for i, subgoal_emb in enumerate(subgoal_embeddings):
                reconstructed_state = self.embedding_model.decode(subgoal_emb)
                
                # Convert tensor back to state format expected by environment
                state_array = reconstructed_state.squeeze(0).cpu().numpy()
                
                # Create GeneratedSubgoal object
                generated_subgoal = GeneratedSubgoal(
                    subgoal_state=state_array,
                    subgoal_distance=self.default_k_steps,
                    subgoal_id=i,
                    generation_metadata={
                        'embedding_based': True,
                        'embedding_dim': subgoal_emb.shape[-1],
                        'k_steps': self.default_k_steps,
                    }
                )
                subgoals.append(generated_subgoal)
                
        return subgoals


class AdaptiveEmbeddingSubgoalGenerator(InferenceComponent):
    """Adaptive subgoal generator using multiple embedding generators.
    
    Similar to AdaptiveSubgoalGenerator but operates in embedding space
    with different k values for hierarchical planning.
    """
    
    def __init__(
        self,
        generator_k_list: list[int],
        paths_to_generator_weights: list[str],
        path_to_embedding_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: Optional[dict[str, int]] = None,
        embedding_model_class: type[nn.Module] = None,
        generator_class: type[nn.Module] = EmbeddingGenerator,
    ) -> None:
        self.env = env
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.generator_k_list = generator_k_list
        self.path_to_embedding_weights = path_to_embedding_weights
        
        # Create embedding subgoal generators for each k value
        self.subgoal_generators = {
            k: EmbeddingSubgoalGenerator(
                embedding_generator=generator_class,
                state_embedding_model=embedding_model_class,
                path_to_generator_weights=path,
                path_to_embedding_weights=path_to_embedding_weights,
                env=env,
                subgoal_generation_kwargs={
                    **(subgoal_generation_kwargs or {}),
                    'k_steps': k
                }
            )
            for k, path in zip(generator_k_list, paths_to_generator_weights)
        }
        
    def construct_network(self) -> None:
        """Construct all generator networks."""
        for generator in self.subgoal_generators.values():
            generator.construct_network()
            
    def get_network(self) -> dict[str, dict[str, nn.Module]]:
        """Return all networks organized by k value."""
        return {
            str(k): generator.get_network() 
            for k, generator in self.subgoal_generators.items()
        }
        
    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """Generate subgoals using the appropriate k-step generator.
        
        Uses the k value specified in the node or defaults to the first one.
        """
        # Determine which generator to use
        k_value = getattr(node, 'next_expand_with_k_generator', self.generator_k_list[0])
        
        if k_value not in self.subgoal_generators:
            # Fall back to closest available k value
            k_value = min(self.generator_k_list, key=lambda x: abs(x - k_value))
            
        generator = self.subgoal_generators[k_value]
        return generator.get_subgoals(node)


class HybridSubgoalGenerator(InferenceComponent):
    """Hybrid generator that can use both discrete and embedding-based generation.
    
    Provides compatibility with existing discrete generators while enabling
    embedding-based generation for improved performance.
    """
    
    def __init__(
        self,
        discrete_generator: SubgoalGenerator,
        embedding_generator: EmbeddingSubgoalGenerator,
        use_embedding_ratio: float = 0.5,
    ) -> None:
        self.discrete_generator = discrete_generator
        self.embedding_generator = embedding_generator
        self.use_embedding_ratio = use_embedding_ratio
        
    def construct_network(self) -> None:
        """Construct both generator networks."""
        self.discrete_generator.construct_network()
        self.embedding_generator.construct_network()
        
    def get_network(self) -> dict[str, nn.Module]:
        """Return networks from both generators."""
        discrete_net = self.discrete_generator.get_network()
        embedding_net = self.embedding_generator.get_network()
        
        if isinstance(discrete_net, dict) and isinstance(embedding_net, dict):
            return {**discrete_net, **embedding_net}
        else:
            return {
                'discrete': discrete_net,
                'embedding': embedding_net
            }
            
    def get_subgoals(self, node: SearchTreeNode) -> list[GeneratedSubgoal]:
        """Generate subgoals using both methods and combine them."""
        # Generate subgoals from both generators
        discrete_subgoals = self.discrete_generator.get_subgoals(node)
        embedding_subgoals = self.embedding_generator.get_subgoals(node)
        
        # Combine subgoals based on ratio
        total_subgoals = len(discrete_subgoals) + len(embedding_subgoals)
        num_embedding = int(total_subgoals * self.use_embedding_ratio)
        num_discrete = total_subgoals - num_embedding
        
        # Select subgoals
        selected_discrete = discrete_subgoals[:num_discrete]
        selected_embedding = embedding_subgoals[:num_embedding]
        
        # Mark subgoals with their generation method
        for subgoal in selected_discrete:
            subgoal.generation_metadata = {
                **(subgoal.generation_metadata or {}),
                'generation_method': 'discrete'
            }
            
        for subgoal in selected_embedding:
            subgoal.generation_metadata = {
                **(subgoal.generation_metadata or {}),
                'generation_method': 'embedding'
            }
            
        return selected_discrete + selected_embedding