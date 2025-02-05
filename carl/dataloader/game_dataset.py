from torch import Tensor
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, boards: Tensor, targets: Tensor) -> None:
        self.boards = boards
        self.targets = targets

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        board: Tensor = self.boards[idx]
        target: Tensor = self.targets[idx]
        return board, target
