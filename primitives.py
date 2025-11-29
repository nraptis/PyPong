# primitives.py

from dataclasses import dataclass
from abc import ABC, abstractmethod
from float_bufferable import FloatBufferable

class ColorConforming(ABC):
    @property
    @abstractmethod
    def r(self) -> float: ...
    @r.setter
    @abstractmethod
    def r(self, value: float): ...

    @property
    @abstractmethod
    def g(self) -> float: ...
    @g.setter
    @abstractmethod
    def g(self, value: float): ...

    @property
    @abstractmethod
    def b(self) -> float: ...
    @b.setter
    @abstractmethod
    def b(self, value: float): ...

    @property
    @abstractmethod
    def a(self) -> float: ...
    @a.setter
    @abstractmethod
    def a(self, value: float): ...


class PositionConforming2D(ABC):
    @property
    @abstractmethod
    def x(self) -> float: ...
    @x.setter
    @abstractmethod
    def x(self, value: float): ...

    @property
    @abstractmethod
    def y(self) -> float: ...
    @y.setter
    @abstractmethod
    def y(self, value: float): ...


class TextureCoordinateConforming(ABC):
    @property
    @abstractmethod
    def u(self) -> float: ...
    @u.setter
    @abstractmethod
    def u(self, value: float): ...

    @property
    @abstractmethod
    def v(self) -> float: ...
    @v.setter
    @abstractmethod
    def v(self, value: float): ...

@dataclass
class Shape2DVertex(PositionConforming2D, FloatBufferable):
    x: float = 0.0
    y: float = 0.0

    def write_to_buffer(self, buffer):
        buffer.append(self.x)
        buffer.append(self.y)

    def size(self):
        return 2

@dataclass
class Shape2DColoredVertex(PositionConforming2D, ColorConforming, FloatBufferable):
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
class Sprite2DVertex(PositionConforming2D, TextureCoordinateConforming, FloatBufferable):
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
    FloatBufferable
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
