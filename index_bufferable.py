#index_bufferable.py

from abc import ABC, abstractmethod

class IndexBufferable(ABC):

    @abstractmethod
    def write_to_buffer(self, buffer) -> None:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError