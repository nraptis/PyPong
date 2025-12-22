# float_bufferable.py

from __future__ import annotations
from typing import Protocol

class GraphicsFloatBufferable(Protocol):
    """
    Any object that can write its float data into a buffer.
    """

    def write_to_buffer(self, buffer) -> None:
        ...

    def size(self) -> int:
        ...
