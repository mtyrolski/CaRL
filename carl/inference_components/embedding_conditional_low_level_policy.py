"""Embedding-based Conditional Low-Level Policy for CaRL inference framework.

This module provides CLLPs that operate on state embeddings rather than
explicit states, enabling hierarchical navigation in latent space.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.inference_components.component import InferenceComponent
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.components.state_embeddings import BaseStateEmbedding
from carl.components.embedding_cllp import EmbeddingConditionedCLLP


class EmbeddingConditionalLowLevelPolicy(ConditionalLowLevelPolicy):
    """Conditional Low-Level Policy that operates in embedding space.
    
    Uses pretrained state embedding model and embedding-conditioned CLLP
    to perform hierarchical navigation in latent space.
    """
    
    def __init__(
        self,
        cllp_network_class: type[nn.Module],
        embedding_model_class: type[nn.Module],
        path_to_cllp_weights: str,
        path_to_embedding_weights: str,
        env: GameEnv,
        num_actions_to_generate: int = 50,
    ) -> None:
        super().__init__(
            cllp_network_class,
            path_to_cllp_weights,
            env,
            num_actions_to_generate
        )
        
        self.embedding_model_class = embedding_model_class
        self.path_to_embedding_weights = path_to_embedding_weights
        self.embedding_model: Optional[BaseStateEmbedding] = None
        self.cllp_model: Optional[EmbeddingConditionedCLLP] = None
        
    def construct_network(self) -> None:
        """Construct both embedding and CLLP networks."""
        # Load state embedding model
        self.embedding_model = self.instantiate_network(
            self.embedding_model_class,
            self.path_to_embedding_weights
        )
        
        # Load embedding CLLP model
        self.cllp_model = self.instantiate_network(
            self.cllp_network_class,
            self.path_to_cllp_weights
        )
        
    def get_network(self) -> dict[str, nn.Module]:
        """Return both networks."""
        assert self.embedding_model is not None, "Embedding model not constructed"
        assert self.cllp_model is not None, "CLLP model not constructed"
        
        return {
            'embedding_model': self.embedding_model,
            'cllp_model': self.cllp_model
        }
        
    def get_actions(
        self, 
        current_state: np.ndarray, 
        subgoal_state: np.ndarray
    ) -> list[int]:
        """Generate actions to navigate from current state to subgoal in embedding space.
        
        Args:
            current_state: Current state as array
            subgoal_state: Target subgoal state as array
            
        Returns:
            List of actions to reach subgoal
        """
        assert self.embedding_model is not None, "Embedding model not constructed"
        assert self.cllp_model is not None, "CLLP model not constructed"
        
        actions = []
        current = current_state.copy()
        
        # Convert states to embeddings
        current_tensor = self._state_to_tensor(current)
        subgoal_tensor = self._state_to_tensor(subgoal_state)
        
        with torch.no_grad():
            # Encode states to embeddings
            current_embedding = self.embedding_model.encode(current_tensor)
            subgoal_embedding = self.embedding_model.encode(subgoal_tensor)
            
            # Generate actions using CLLP in embedding space
            for step in range(self.num_actions_to_generate):
                # Check if we've reached the subgoal
                if self.env.is_same_state(current, subgoal_state):
                    break
                    
                # Get action probabilities from embedding CLLP
                action_logits = self.cllp_model(current_embedding, subgoal_embedding)
                action_probs = torch.softmax(action_logits, dim=-1)
                
                # Sample action (or take most likely)
                action = torch.argmax(action_probs, dim=-1).item()
                actions.append(action)
                
                # Apply action to get next state
                next_state = self.env.apply_action(current, action)
                if next_state is None or self.env.is_same_state(next_state, current):
                    # Invalid action or no progress, stop
                    break
                    
                current = next_state
                
                # Update current embedding for next iteration
                current_tensor = self._state_to_tensor(current)
                current_embedding = self.embedding_model.encode(current_tensor)
                
        return actions
        
    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert state array to tensor for embedding model."""
        # Use tokenizer to convert state to proper tensor format
        state_tensor, _ = self.env.tokenizer.x_y_tokenizer(
            state, state, 'state_embedding_ae'
        )
        return state_tensor.unsqueeze(0)  # Add batch dimension


class ProgressAwareEmbeddingCLLP(EmbeddingConditionalLowLevelPolicy):
    """Embedding CLLP with progress tracking towards subgoals.
    
    Monitors progress in embedding space and can provide early termination
    or adaptive action selection based on progress estimates.
    """
    
    def __init__(
        self,
        cllp_network_class: type[nn.Module],
        embedding_model_class: type[nn.Module], 
        path_to_cllp_weights: str,
        path_to_embedding_weights: str,
        env: GameEnv,
        num_actions_to_generate: int = 50,
        progress_threshold: float = 0.9,
    ) -> None:
        super().__init__(
            cllp_network_class,
            embedding_model_class,
            path_to_cllp_weights,
            path_to_embedding_weights,
            env,
            num_actions_to_generate
        )
        self.progress_threshold = progress_threshold
        
    def get_actions(
        self,
        current_state: np.ndarray,
        subgoal_state: np.ndarray
    ) -> list[int]:
        """Generate actions with progress monitoring."""
        assert self.embedding_model is not None, "Embedding model not constructed"
        assert self.cllp_model is not None, "CLLP model not constructed"
        
        actions = []
        current = current_state.copy()
        
        # Convert states to embeddings
        current_tensor = self._state_to_tensor(current)
        subgoal_tensor = self._state_to_tensor(subgoal_state)
        
        with torch.no_grad():
            # Encode states to embeddings
            current_embedding = self.embedding_model.encode(current_tensor)
            subgoal_embedding = self.embedding_model.encode(subgoal_tensor)
            initial_embedding = current_embedding.clone()
            
            # Calculate initial distance in embedding space
            initial_distance = torch.norm(current_embedding - subgoal_embedding).item()
            
            for step in range(self.num_actions_to_generate):
                # Check if we've reached the subgoal
                if self.env.is_same_state(current, subgoal_state):
                    break
                    
                # Calculate progress in embedding space
                current_distance = torch.norm(current_embedding - subgoal_embedding).item()
                progress = 1.0 - (current_distance / (initial_distance + 1e-8))
                
                # Early termination if sufficient progress made
                if progress >= self.progress_threshold:
                    break
                    
                # Get action from CLLP (with progress if model supports it)
                if hasattr(self.cllp_model, 'forward') and len(torch.nn.utils.parameters_to_vector(self.cllp_model.parameters()).shape) > 1:
                    # Check if model returns progress estimate
                    try:
                        action_logits, predicted_progress = self.cllp_model(current_embedding, subgoal_embedding)
                    except (ValueError, TypeError):
                        action_logits = self.cllp_model(current_embedding, subgoal_embedding)
                else:
                    action_logits = self.cllp_model(current_embedding, subgoal_embedding)
                
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
                actions.append(action)
                
                # Apply action and update state
                next_state = self.env.apply_action(current, action)
                if next_state is None or self.env.is_same_state(next_state, current):
                    break
                    
                current = next_state
                current_tensor = self._state_to_tensor(current)
                current_embedding = self.embedding_model.encode(current_tensor)
                
        return actions


class MultiScaleEmbeddingCLLP(EmbeddingConditionalLowLevelPolicy):
    """CLLP that operates at multiple scales in embedding space.
    
    Can handle both short-term and long-term subgoals by using different
    embedding representations or hierarchical approaches.
    """
    
    def __init__(
        self,
        cllp_network_class: type[nn.Module],
        embedding_model_class: type[nn.Module],
        path_to_cllp_weights: str,
        path_to_embedding_weights: str,
        env: GameEnv,
        num_actions_to_generate: int = 50,
        scale_factors: list[float] = [1.0, 0.5, 0.25],
    ) -> None:
        super().__init__(
            cllp_network_class,
            embedding_model_class,
            path_to_cllp_weights,
            path_to_embedding_weights,
            env,
            num_actions_to_generate
        )
        self.scale_factors = scale_factors
        
    def get_actions(
        self,
        current_state: np.ndarray,
        subgoal_state: np.ndarray
    ) -> list[int]:
        """Generate actions considering multiple scales."""
        assert self.embedding_model is not None, "Embedding model not constructed"
        assert self.cllp_model is not None, "CLLP model not constructed"
        
        actions = []
        current = current_state.copy()
        
        # Convert states to embeddings
        current_tensor = self._state_to_tensor(current)
        subgoal_tensor = self._state_to_tensor(subgoal_state)
        
        with torch.no_grad():
            current_embedding = self.embedding_model.encode(current_tensor)
            subgoal_embedding = self.embedding_model.encode(subgoal_tensor)
            
            for step in range(self.num_actions_to_generate):
                if self.env.is_same_state(current, subgoal_state):
                    break
                    
                # Aggregate action probabilities across scales
                total_action_probs = None
                
                for scale in self.scale_factors:
                    # Scale the embedding difference
                    scaled_subgoal = current_embedding + scale * (subgoal_embedding - current_embedding)
                    
                    # Get action probabilities for this scale
                    action_logits = self.cllp_model(current_embedding, scaled_subgoal)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    
                    if total_action_probs is None:
                        total_action_probs = action_probs
                    else:
                        total_action_probs += action_probs
                        
                # Normalize and select action
                total_action_probs = total_action_probs / len(self.scale_factors)
                action = torch.argmax(total_action_probs, dim=-1).item()
                actions.append(action)
                
                # Apply action and update
                next_state = self.env.apply_action(current, action)
                if next_state is None or self.env.is_same_state(next_state, current):
                    break
                    
                current = next_state
                current_tensor = self._state_to_tensor(current)
                current_embedding = self.embedding_model.encode(current_tensor)
                
        return actions


class EmbeddingCLLPValidator(InferenceComponent):
    """Validator that uses embedding CLLP to check subgoal reachability.
    
    Integrates with CaRL's validation framework to assess whether
    subgoals generated in embedding space are actually reachable.
    """
    
    def __init__(
        self,
        env: GameEnv,
        embedding_cllp: EmbeddingConditionalLowLevelPolicy,
        max_validation_steps: int = 100,
        success_threshold: float = 0.1,  # Threshold for considering subgoal reached
    ) -> None:
        self.env = env
        self.embedding_cllp = embedding_cllp
        self.max_validation_steps = max_validation_steps
        self.success_threshold = success_threshold
        
    def construct_network(self) -> None:
        """Construct the CLLP network."""
        self.embedding_cllp.construct_network()
        
    def get_network(self) -> dict[str, nn.Module]:
        """Return CLLP networks."""
        return self.embedding_cllp.get_network()
        
    def is_valid(
        self, 
        state: np.ndarray, 
        subgoal: np.ndarray, 
        steps_limit: Optional[int] = None
    ) -> dict:
        """Validate if subgoal is reachable using embedding CLLP.
        
        Returns:
            Dictionary with validation results including success, distance, actions
        """
        if steps_limit is None:
            steps_limit = self.max_validation_steps
            
        # Try to reach subgoal using embedding CLLP
        actions = self.embedding_cllp.get_actions(state, subgoal)
        
        # Simulate execution of actions
        current_state = state.copy()
        executed_actions = []
        
        for i, action in enumerate(actions[:steps_limit]):
            next_state = self.env.apply_action(current_state, action)
            if next_state is None:
                break
                
            executed_actions.append(action)
            current_state = next_state
            
            # Check if we've reached the subgoal (with some tolerance)
            if self._states_close(current_state, subgoal):
                return {
                    'success': True,
                    'is_solved': self.env.is_solved(current_state),
                    'actions': executed_actions,
                    'steps': len(executed_actions),
                    'final_state': current_state,
                    'distance_to_subgoal': self._state_distance(current_state, subgoal)
                }
                
        # Failed to reach subgoal
        return {
            'success': False,
            'is_solved': self.env.is_solved(current_state),
            'actions': executed_actions,
            'steps': len(executed_actions),
            'final_state': current_state,
            'distance_to_subgoal': self._state_distance(current_state, subgoal)
        }
        
    def _states_close(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if two states are close enough to be considered the same."""
        if self.env.is_same_state(state1, state2):
            return True
            
        # Use embedding distance as fallback
        try:
            embedding_model = self.embedding_cllp.embedding_model
            if embedding_model is not None:
                with torch.no_grad():
                    emb1 = embedding_model.encode(self.embedding_cllp._state_to_tensor(state1))
                    emb2 = embedding_model.encode(self.embedding_cllp._state_to_tensor(state2))
                    distance = torch.norm(emb1 - emb2).item()
                    return distance < self.success_threshold
        except Exception:
            pass
            
        return False
        
    def _state_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute distance between states (in embedding space if possible)."""
        try:
            embedding_model = self.embedding_cllp.embedding_model
            if embedding_model is not None:
                with torch.no_grad():
                    emb1 = embedding_model.encode(self.embedding_cllp._state_to_tensor(state1))
                    emb2 = embedding_model.encode(self.embedding_cllp._state_to_tensor(state2))
                    return torch.norm(emb1 - emb2).item()
        except Exception:
            pass
            
        # Fallback to L2 distance in state space
        return np.linalg.norm(state1.flatten() - state2.flatten())