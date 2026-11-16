"""Interfaces for experiment loggers."""
from abc import ABC, abstractmethod
from typing import Any


class CaRLLogger(ABC):
    """Create a callback usable by a CaRL training or evaluation run."""

    @abstractmethod
    def return_logger(self) -> Any:
        raise NotImplementedError
