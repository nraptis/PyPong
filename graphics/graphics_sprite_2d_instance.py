# graphics_sprite_2d_instance.py

from graphics.graphics_sprite_instance import GraphicsSpriteInstance
from graphics.graphics_primitives import Sprite2DVertex

class GraphicsSprite2DInstance(GraphicsSpriteInstance[Sprite2DVertex]):
    def __init__(self) -> None:
        vertex_array = [
            Sprite2DVertex(0.0,   0.0,   0.0, 0.0),
            Sprite2DVertex(256.0, 0.0,   1.0, 0.0),
            Sprite2DVertex(0.0,   256.0, 0.0, 1.0),
            Sprite2DVertex(256.0, 256.0, 1.0, 1.0),
        ]
        super().__init__(vertex_array)