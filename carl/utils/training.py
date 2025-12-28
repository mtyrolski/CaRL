import torch
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel

from carl.dataloader.game_dataset import GameDataset
from carl.inference_components.component import InferenceComponent
from carl.inference_components.component import TrainingModule
from carl.inference_components.conditional_low_level_policy import TransformerConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import AdaptiveSubgoalGenerator
from carl.inference_components.subgoal_generator import TransformerSubgoalGenerator
from carl.inference_components.value import TransformerValue
from carl.memory.replay_buffer import OfflineReplayBuffer
from carl.memory.replay_buffer import SimpleUniversalReplayBuffer


def prepare_data_for_training_component(data_x_y: list[torch.Tensor],
                                        test_size: float) -> tuple[GameDataset, GameDataset]:
    """
    Given a list of data points, split the data into training and validation datasets.
    """
    train_data_x_y: list[tuple[torch.Tensor, torch.Tensor]]
    validation_data_x_y: list[tuple[torch.Tensor, torch.Tensor]]

    train_data_x_y, validation_data_x_y = train_test_split(data_x_y, test_size=test_size)

    x_train: torch.Tensor = torch.cat([x for x, _ in train_data_x_y])
    y_train: torch.Tensor = torch.cat([y for _, y in train_data_x_y])
    x_validation: torch.Tensor = torch.cat([x for x, _ in validation_data_x_y])
    y_validation: torch.Tensor = torch.cat([y for _, y in validation_data_x_y])

    train_dataset: GameDataset = GameDataset(x_train, y_train)
    validation_dataset: GameDataset = GameDataset(x_validation, y_validation)

    return train_dataset, validation_dataset


def is_hierarchical_component(component: InferenceComponent) -> bool:
    """
    Check if the component is hierarchical.
    """
    training_module = component.get_component_training_module()
    assert isinstance(training_module, (TrainingModule, dict))
    return isinstance(training_module, dict)


BufferContainer = OfflineReplayBuffer | dict[int, OfflineReplayBuffer] | dict[str, OfflineReplayBuffer]
NetworkContainer = PreTrainedModel | dict[int, PreTrainedModel] | dict[str, PreTrainedModel]


def get_buffer_for_training(component: InferenceComponent,
                            reply_buffer: SimpleUniversalReplayBuffer) -> BufferContainer:
    """
    Extract buffer for training.
    """
    if isinstance(component, TransformerSubgoalGenerator):
        buffer = reply_buffer.get_buffer_for_generator(None)
        assert isinstance(buffer, OfflineReplayBuffer)
        return buffer
    if isinstance(component, TransformerValue):
        return reply_buffer.get_buffer_for_value()
    if isinstance(component, TransformerConditionalLowLevelPolicy):
        return reply_buffer.get_buffer_for_policy()
    if isinstance(component, AdaptiveSubgoalGenerator):
        buffers: dict[int, OfflineReplayBuffer] = {}
        for k in component.generator_k_list:
            buffer = reply_buffer.get_buffer_for_generator(k)
            assert isinstance(buffer, OfflineReplayBuffer)
            buffers[k] = buffer
        return buffers

    raise ValueError('Component does not have a buffer for training.')


def iterate_networks_for_training(
        component: InferenceComponent,
        reply_buffer: SimpleUniversalReplayBuffer) -> tuple[NetworkContainer, BufferContainer]:
    """
    Extract networks and buffers for training.
    """
    network_container = component.get_network()
    buffer_container = get_buffer_for_training(component, reply_buffer)

    # Validation
    variant1: bool = isinstance(network_container, PreTrainedModel) and isinstance(buffer_container,
                                                                                   OfflineReplayBuffer)
    variant2: bool = isinstance(network_container, dict) and isinstance(buffer_container, dict) and all(
        isinstance(v, PreTrainedModel) for v in network_container.values()) and all(
            isinstance(v, OfflineReplayBuffer) for v in buffer_container.values())

    if not (variant1 or variant2):
        raise ValueError('Invalid network and buffer containers.')

    return network_container, buffer_container
