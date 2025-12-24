# paddle.py
from __future__ import annotations
from typing import Optional
from pong.pong_asset_bundle import PongAssetBundle
from graphics.graphics_library import GraphicsLibrary
from graphics.graphics_sprite import GraphicsSprite
from graphics.graphics_pipeline import GraphicsPipeline
from graphics.graphics_matrix import GraphicsMatrix
from graphics.graphics_sprite_2d_instance import GraphicsSprite2DInstance
from graphics.graphics_color import GraphicsColor

class PongPaddle:
    def __init__(
        self,
        x: float,
        y: float
    ) -> None:
        # Logical state
        self.x = x
        self.y = y
        self.instance = GraphicsSprite2DInstance()
        self.width = float(PongAssetBundle.paddle_width)
        self.height = float(PongAssetBundle.paddle_height)
        self.is_red = False

    # ------------------------------------------------------------------
    # Lifecycle: load
    # ------------------------------------------------------------------
    def load(self, assets: Optional[PongAssetBundle], graphics: Optional[GraphicsLibrary]) -> None:
        self.instance.load(graphics=graphics, sprite=assets.paddle_sprite)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        pass

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def draw(self,
             graphics: Optional[GraphicsLibrary],
             pipeline: Optional[GraphicsPipeline],
             projection_matrix: Optional[GraphicsMatrix]) -> None:
        
        if graphics is None:
            return
        if pipeline is None:
            return
        if projection_matrix is None:
            return
        
        model_view_matrix = GraphicsMatrix()
        model_view_matrix.translation(x=self.x, y=self.y, z=0.0)
        self.instance.projection_matrix = projection_matrix
        self.instance.model_view_matrix = model_view_matrix
        if self.is_red:
            self.instance.color = GraphicsColor(1.0, 0.25, 0.25, 1.0)
        else:
            self.instance.color = GraphicsColor(1.0, 1.0, 1.0, 1.0)

        self.instance.render(shader_program=pipeline.program_sprite_2d)
    
    def dispose(self) -> None:
        self.instance.dispose()