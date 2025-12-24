# pong_number.py

from __future__ import annotations
from typing import List

from graphics.graphics_library import GraphicsLibrary
from graphics.graphics_pipeline import GraphicsPipeline
from graphics.graphics_matrix import GraphicsMatrix
from graphics.graphics_sprite_2d_instance import GraphicsSprite2DInstance
from pong.pong_asset_bundle import PongAssetBundle
from graphics.graphics_texture import GraphicsTexture
from graphics.shader_program import ShaderProgram

class PongNumber:
    # Horizontal spacing per digit in screen units
    digit_width: float = 60.0

    
    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0

        # One sprite instance per digit
        self.digit_instances: List[GraphicsSprite2DInstance] = []

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------
    def rebuild(self, score: int, graphics: GraphicsLibrary, assets: PongAssetBundle) -> None:
        self.dispose()
        value = int(score)
        string = str(value)
        for character in string:
            digit = int(character)
            sprite = assets.get_digit_sprite(digit)
            instance = GraphicsSprite2DInstance()
            instance.load(graphics, sprite)
            self.digit_instances.append(instance)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def draw(
        self,
        graphics: GraphicsLibrary,
        pipeline: GraphicsPipeline,
        projection_matrix: GraphicsMatrix,
    ) -> None:
        if not self.digit_instances:
            return
        
        digit_count = len(self.digit_instances)
        total_width = digit_count * PongNumber.digit_width
        offset_x = (-total_width) * 0.5
        for i, instance in enumerate(self.digit_instances):
            instance.projection_matrix = projection_matrix.copy()
            instance.model_view_matrix.translation(self.x + offset_x, self.y, 0.0)
            instance.render(pipeline.program_sprite_2d)
            offset_x += float(PongNumber.digit_width)

    # ------------------------------------------------------------------
    # Dispose
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        for instance in self.digit_instances:
            instance.dispose()
        self.digit_instances.clear()
