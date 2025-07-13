from abc import ABC
from abc import abstractmethod


class Algorithm(ABC):
    """Abstract base class for algorithms.
    This class defines the interface for all algorithms in the CARL framework.
    It requires the implementation of just run method so it is easily runnable using hydra-config.

    Args:
        ABC (type): Abstract base class for defining abstract methods.
    """
    
    @abstractmethod
    def run(self) -> None:
        pass
