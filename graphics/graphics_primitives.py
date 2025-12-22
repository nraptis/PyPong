# graphics_primitives.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from graphics.graphics_float_bufferable import GraphicsFloatBufferable

# -------------------------------------------------------------
# Protocols
# -------------------------------------------------------------

class ColorConforming(Protocol):
    @property
    def r(self) -> float: ...
    @r.setter
    def r(self, value: float) -> None: ...

    @property
    def g(self) -> float: ...
    @g.setter
    def g(self, value: float) -> None: ...

    @property
    def b(self) -> float: ...
    @b.setter
    def b(self, value: float) -> None: ...

    @property
    def a(self) -> float: ...
    @a.setter
    def a(self, value: float) -> None: ...


class PositionConforming2D(Protocol):
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, value: float) -> None: ...

    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, value: float) -> None: ...


class TextureCoordinateConforming(Protocol):
    @property
    def u(self) -> float: ...
    @u.setter
    def u(self, value: float) -> None: ...

    @property
    def v(self) -> float: ...
    @v.setter
    def v(self, value: float) -> None: ...


# -------------------------------------------------------------
# Concrete vertex dataclasses (auto-conform via structural typing)
# -------------------------------------------------------------

@dataclass
class Shape2DVertex(PositionConforming2D, GraphicsFloatBufferable):
    x: float = 0.0
    y: float = 0.0

    def write_to_buffer(self, buffer):
        buffer.append(self.x)
        buffer.append(self.y)

    def size(self):
        return 2

@dataclass
class Shape2DColoredVertex(
    PositionConforming2D,
    ColorConforming,
    GraphicsFloatBufferable
):
    x: float = 0.0
    y: float = 0.0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def write_to_buffer(self, buffer):
        buffer.append(self.x)
        buffer.append(self.y)
        buffer.append(self.r)
        buffer.append(self.g)
        buffer.append(self.b)
        buffer.append(self.a)

    def size(self):
        return 6


@dataclass
class Sprite2DVertex(
    PositionConforming2D,
    TextureCoordinateConforming,
    GraphicsFloatBufferable
):
    x: float = 0.0
    y: float = 0.0
    u: float = 0.0
    v: float = 0.0

    def write_to_buffer(self, buffer):
        buffer.append(self.x)
        buffer.append(self.y)
        buffer.append(self.u)
        buffer.append(self.v)

    def size(self):
        return 4


@dataclass
class Sprite2DColoredVertex(
    PositionConforming2D,
    TextureCoordinateConforming,
    ColorConforming,
    GraphicsFloatBufferable
):
    x: float = 0.0
    y: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def write_to_buffer(self, buffer):
        buffer.append(self.x)
        buffer.append(self.y)
        buffer.append(self.u)
        buffer.append(self.v)
        buffer.append(self.r)
        buffer.append(self.g)
        buffer.append(self.b)
        buffer.append(self.a)

    def size(self):
        return 8
