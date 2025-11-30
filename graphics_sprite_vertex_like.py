# graphics_sprite_vertex_like.py
from __future__ import annotations
from typing import Protocol

class GraphicsSpriteVertexLike(Protocol):
    """
    A vertex suitable for 2D sprites:
      - has x, y (position)
      - has u, v (texture coordinates)
      - can write itself to a float buffer
    """

    x: float
    y: float
    u: float
    v: float

    def write_to_buffer(self, buffer) -> None:
        ...

    def size(self) -> int:
        ...
