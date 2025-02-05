from abc import ABC
from abc import abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def run(self) -> None:
        pass
