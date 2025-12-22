# graphics_shape_2d_instance.py

from graphics.graphics_shape_instance import GraphicsShapeInstance
from graphics.graphics_primitives import Shape2DVertex

class GraphicsShape2DInstance(GraphicsShapeInstance[Shape2DVertex]):
    def __init__(self) -> None:
        vertex_array = [
            Shape2DVertex(0.0,   0.0),
            Shape2DVertex(256.0, 0.0),
            Shape2DVertex(0.0,   256.0),
            Shape2DVertex(256.0, 256.0),
        ]
        super().__init__(vertex_array)
