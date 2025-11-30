# asset_bundle.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

from graphics_texture import GraphicsTexture
from graphics_sprite import GraphicsSprite
from graphics_library import GraphicsLibrary

class AssetBundle:

    # class-level (static) constants
    paddle_width: int = 48
    paddle_height: int = 228

    # class-level (static) constants
    ball_width: int = 48
    ball_height: int = 48

    def __init__(self) -> None:
        # Sprites
        self.ball_sprite: Optional[GraphicsSprite] = None
        self.paddle_sprite: Optional[GraphicsSprite] = None
        self.digit_table: dict[int, GraphicsSprite] = {}

        # Textures
        self.ball_texture: Optional[GraphicsTexture] = None
        self.paddle_texture: Optional[GraphicsTexture] = None
        self.wall_texture: Optional[GraphicsTexture] = None
        self.digit_textures: list[GraphicsTexture] = []

        self.loaded = False

    # ------------------------------------------------------------------
    # Initial load: create texture + sprite instances
    # ------------------------------------------------------------------
    def load(self, graphics: GraphicsLibrary, base_dir: Path) -> None:
        images = base_dir / "images"

        # --------------------------------------------------------------
        # ball.png → texture + sprite
        # --------------------------------------------------------------
        ball_path = images / "ball.png"
        self.ball_texture = GraphicsTexture(graphics=graphics, file_name=ball_path)
        self.ball_texture.print()

        self.ball_sprite = GraphicsSprite()
        self.ball_sprite.load(graphics=graphics, texture=self.ball_texture)
        self.ball_sprite.print()

        # --------------------------------------------------------------
        # digit_0.png ... digit_9.png → textures + sprites
        # --------------------------------------------------------------
        self.digit_textures = []
        self.digit_table.clear()

        for i in range(10):
            digit_path = images / f"digit_{i}.png"
            tex = GraphicsTexture(graphics=graphics, file_name=digit_path)
            tex.print()
            self.digit_textures.append(tex)

            spr = GraphicsSprite()
            spr.load(graphics=graphics, texture=tex)
            spr.print()

            self.digit_table[i] = spr

        # --------------------------------------------------------------
        # paddle.png → texture + sprite
        # --------------------------------------------------------------
        paddle_path = images / "paddle.png"
        self.paddle_texture = GraphicsTexture(graphics=graphics, file_name=paddle_path)
        self.paddle_texture.print()

        self.paddle_sprite = GraphicsSprite()
        self.paddle_sprite.load(graphics=graphics, texture=self.paddle_texture)
        self.paddle_sprite.print()

        # --------------------------------------------------------------
        # wall.png → texture only
        # --------------------------------------------------------------
        wall_path = images / "wall.png"
        self.wall_texture = GraphicsTexture(graphics=graphics, file_name=wall_path)
        self.wall_texture.print()
        
        self.loaded = True

    # ------------------------------------------------------------------
    # Dispose all textures (sprites and dict stay)
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        """
        Dispose GPU textures. Texture objects and sprites stay alive,
        but their GL handles are released.
        Safe to call multiple times.
        """
        if self.ball_texture is not None:
            self.ball_texture.dispose()

        if self.paddle_texture is not None:
            self.paddle_texture.dispose()

        if self.wall_texture is not None:
            self.wall_texture.dispose()

        for tex in self.digit_textures:
            tex.dispose()

        self.loaded = False

    # ------------------------------------------------------------------
    # Reload: reuse same texture objects, reload GL resources
    # ------------------------------------------------------------------
    def reload(self, graphics: GraphicsLibrary) -> None:
        """
        Reload all textures using existing GraphicsTexture instances.
        Sprites and digit_table are reused.
        """
        # Update graphics on each texture and call load()
        if self.ball_texture is not None:
            self.ball_texture.graphics = graphics
            self.ball_texture.load()

        if self.paddle_texture is not None:
            self.paddle_texture.graphics = graphics
            self.paddle_texture.load()

        if self.wall_texture is not None:
            self.wall_texture.graphics = graphics
            self.wall_texture.load()

        for tex in self.digit_textures:
            tex.graphics = graphics
            tex.load()

        self.loaded = True

    # ------------------------------------------------------------------
    # Digit access
    # ------------------------------------------------------------------
    def get_digit_sprite(self, digit: int) -> GraphicsSprite:
        """
        Return the sprite for a given digit 0–9.
        Clamps out-of-range values into [0, 9].
        """
        d = max(0, min(9, int(digit)))
        return self.digit_table[d]
