# float_bufferable.py

from abc import ABC, abstractmethod

class FloatBufferable(ABC):

    @abstractmethod
    def write_to_buffer(self, buffer) -> None:
        """Write this object's float data into the buffer."""
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        """Return number of floats written."""
        raise NotImplementedError