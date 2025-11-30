# float_bufferable.py

from abc import ABC, abstractmethod

class FloatBufferable(ABC):
    
    @abstractmethod
    def write_to_buffer(self, buffer) -> None:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError