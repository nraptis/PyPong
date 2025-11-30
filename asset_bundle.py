# asset_bundle.py
from __future__ import annotations
from pathlib import Path

from graphics_texture import GraphicsTexture
from graphics_sprite import GraphicsSprite

class AssetBundle:
    def __init__(self) -> None:
        # Sprites
        self.ball_sprite: GraphicsSprite | None = None
        self.paddle_sprite: GraphicsSprite | None = None
        self.digit_sprites: dict[int, GraphicsSprite] = {}

        # Textures
        self.wall_texture: GraphicsTexture | None = None

    # ------------------------------------------------------------------
    # Load all assets
    # ------------------------------------------------------------------
    def load(self, graphics, base_dir: Path) -> None:
        
        images = base_dir / "images"

        # --------------------------------------------------------------
        # ball.png → sprite
        # --------------------------------------------------------------
        ball_path = images / "ball.png"
        ball_tex = GraphicsTexture(graphics=graphics, file_name=ball_path)
        ball_tex.print()

        ball_sprite = GraphicsSprite()
        ball_sprite.load(graphics=graphics, texture=ball_tex)
        ball_sprite.print()
        self.ball_sprite = ball_sprite

        # --------------------------------------------------------------
        # digit_0.png ... digit_9.png → sprites
        # --------------------------------------------------------------
        for i in range(10):
            digit_path = images / f"digit_{i}.png"
            tex = GraphicsTexture(graphics=graphics, file_name=digit_path)
            tex.print()

            spr = GraphicsSprite()
            spr.load(graphics=graphics, texture=tex)
            spr.print()

            self.digit_sprites[i] = spr

        # --------------------------------------------------------------
        # paddle.png → sprite
        # --------------------------------------------------------------
        paddle_path = images / "paddle.png"
        paddle_tex = GraphicsTexture(graphics=graphics, file_name=paddle_path)
        paddle_tex.print()

        paddle_sprite = GraphicsSprite()
        paddle_sprite.load(graphics=graphics, texture=paddle_tex)
        paddle_sprite.print()
        self.paddle_sprite = paddle_sprite

        # --------------------------------------------------------------
        # wall.png → texture only
        # --------------------------------------------------------------
        wall_path = images / "wall.png"
        wall_tex = GraphicsTexture(graphics=graphics, file_name=wall_path)
        wall_tex.print()

        self.wall_texture = wall_tex

    def get_digit_sprite(self, digit: int) -> GraphicsSprite:
        d = max(0, min(9, int(digit)))
        return self.digit_sprites[d]
