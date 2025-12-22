# color.py

import numpy as np
from graphics.graphics_float_bufferable import GraphicsFloatBufferable
from graphics.graphics_primitives import ColorConforming

class GraphicsColor(GraphicsFloatBufferable, ColorConforming):
    def __init__(self,
                 r: float = 1.0,
                 g: float = 1.0,
                 b: float = 1.0,
                 a: float = 1.0):
        self._r = float(r)
        self._g = float(g)
        self._b = float(b)
        self._a = float(a)

    # -----------------------------
    # ColorConforming properties
    # -----------------------------

    @property
    def r(self) -> float:
        return self._r

    @r.setter
    def r(self, value: float):
        self._r = float(value)

    @property
    def g(self) -> float:
        return self._g

    @g.setter
    def g(self, value: float):
        self._g = float(value)

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, value: float):
        self._b = float(value)

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, value: float):
        self._a = float(value)

    # -----------------------------
    # GraphicsFloatBufferable
    # -----------------------------

    def write_to_buffer(self, buffer):
        buffer.extend([self._r, self._g, self._b, self._a])

    def size(self) -> int:
        return 4

    # -----------------------------
    # Helpers
    # -----------------------------

    def to_array(self) -> np.ndarray:
        return np.array([self._r, self._g, self._b, self._a], dtype=np.float32)

    def copy(self) -> "GraphicsColor":
        return GraphicsColor(self._r, self._g, self._b, self._a)
