# sprite.py

from __future__ import annotations
from typing import Optional
from graphics.graphics_texture import GraphicsTexture

class GraphicsSprite:
    def __init__(self) -> None:
        self.texture: Optional[GraphicsTexture] = None
        self.width: float = 0.0
        self.width2: float = 0.0
        self.height: float = 0.0
        self.height2: float = 0.0
        self.scale_factor: float = 1.0
        self.start_x: float = -64.0
        self.start_y: float = -64.0
        self.end_x: float = 64.0
        self.end_y: float = 64.0
        self.start_u: float = 0.0
        self.start_v: float = 0.0
        self.end_u: float = 1.0
        self.end_v: float = 1.0

    def load(
        self,
        texture: Optional[GraphicsTexture],
        scale_factor: float = 1.0,
    ) -> None:
        if texture is not None:
            self.texture = texture
            self.scale_factor = float(scale_factor)
            self.width = float(texture.width)
            self.height = float(texture.height)
            if self.scale_factor > 1.0:
                self.width = float(int(self.width / self.scale_factor + 0.5))
                self.height = float(int(self.height / self.scale_factor + 0.5))
            self.width2 = float(int(self.width * 0.5 + 0.5))
            self.height2 = float(int(self.height * 0.5 + 0.5))
            _width_2 = -self.width2
            _height_2 = -self.height2
            self.start_x = _width_2
            self.start_y = _height_2
            self.end_x = self.width2
            self.end_y = self.height2
            self.start_u = 0.0
            self.start_v = 0.0
            self.end_u = 1.0
            self.end_v = 1.0
        else:
            self.texture = None
            self.width = 0.0
            self.height = 0.0
            self.width2 = 0.0
            self.height2 = 0.0
            self.scale_factor = 1.0
            
    def set_frame(self, x: float, y: float, width: float, height: float) -> None:
        self.start_x = float(x)
        self.start_y = float(y)
        self.end_x = float(x + width)
        self.end_y = float(y + height)

    def print(self) -> None:
        print("GraphicsSprite -> [" + str(self.width) + ", " + str(self.height) + "]")
        if self.texture:
            print(f"\tGraphicsTexture -> [{self.width}, {self.height}]")
            print(f"\t\tIndex = {self.texture.texture_index}")
            if self.texture.subdirectory is not None or self.texture.name is not None:
                if self.subdirectory is not None and self.texture.name is not None:
                    print(f"\t\tSource = {self.texture.subdirectory}/{self.texture.name}")
                elif self.subdirectory is not None:
                    print(f"\t\tSource = {self.texture.subdirectory}")
                else:
                    print(f"\t\tSource = {self.texture.name}")
                
        print("\tX: [" + str(self.start_x) + ", " + str(self.end_x) + str("]"))
        print("\tY: [" + str(self.start_y) + ", " + str(self.end_y) + str("]"))
        print("\tU: [" + str(self.start_u) + ", " + str(self.end_u) + str("]"))
        print("\tV: [" + str(self.start_v) + ", " + str(self.end_v) + str("]"))
        print("\tScale: " + str(self.scale_factor))
        
