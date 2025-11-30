# paddle.py
from __future__ import annotations
from typing import Optional
from asset_bundle import AssetBundle
from graphics_library import GraphicsLibrary
from graphics_sprite import GraphicsSprite
from graphics_pipeline import GraphicsPipeline
from graphics_matrix import GraphicsMatrix
from graphics_sprite_2d_instance import GraphicsSprite2DInstance

class Paddle:
    def __init__(
        self,
        x: float,
        y: float
    ) -> None:
        # Logical state
        self.x = x
        self.y = y
        self.instance = GraphicsSprite2DInstance()
        self.width = float(AssetBundle.paddle_width)
        self.height = float(AssetBundle.paddle_height)

    # ------------------------------------------------------------------
    # Lifecycle: load
    # ------------------------------------------------------------------
    def load(self, assets: Optional[AssetBundle], graphics: Optional[GraphicsLibrary]) -> None:
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
        self.instance.render(shader_program=pipeline.program_sprite2d)

    # ------------------------------------------------------------------
    # Dispose
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        self.instance.dispose()