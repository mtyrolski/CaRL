"""
Simple Dataset for holding paired board state and target tensors.
"""
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset


class GameDataset(Dataset):
    """
    PyTorch Dataset wrapping input board states and corresponding targets.

    Attributes:
        boards (Tensor): Tensor of input board states, shape (N, ...).
        targets (Tensor): Tensor of target values or labels, shape (N, ...).
    """

    def __init__(self, boards: Tensor, targets: Tensor) -> None:
        """
        Initialize dataset with boards and targets.

        Args:
            boards (Tensor): Tensor of input board states.
            targets (Tensor): Tensor of corresponding targets.
        """
        super().__init__()
        self.boards = boards
        self.targets = targets

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.boards.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: A tuple (board, target) at the given index.
        """
        return self.boards[idx], self.targets[idx]
