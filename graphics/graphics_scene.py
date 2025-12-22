# graphics_scene.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from graphics.graphics_library import GraphicsLibrary
from graphics.graphics_pipeline import GraphicsPipeline

class GraphicsScene(ABC):
    """
    Base class for scenes that work with GraphicsAppShell.

    Each scene is constructed with:
        graphics: GraphicsLibrary
        pipeline: GraphicsPipeline
    """

    def __init__(self, graphics: GraphicsLibrary, pipeline: GraphicsPipeline) -> None:
        # You can replace Any with concrete types if you want, e.g.:
        
        # def __init__(self, graphics: GraphicsLibrary, pipeline: GraphicsPipeline)
        self.graphics = graphics
        self.pipeline = pipeline

    # -------- Lifecycle -------------------------------------------------

    def wake(self) -> None:
        """Called very early before loading."""
        pass

    def load_prepare(self) -> None:
        """Called before load()."""
        pass

    def load(self) -> None:
        """Load textures, buffers, assets, etc."""
        pass

    def load_complete(self) -> None:
        """Called after load()."""
        pass

    def resize(self) -> None:
        """Window or framebuffer size changed."""
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """Advance simulation/game by dt."""
        ...

    @abstractmethod
    def draw(self) -> None:
        """Issue draw calls for this frame."""
        ...

    def dispose(self) -> None:
        """Free GPU resources."""
        pass

    # -------- Input -----------------------------------------------------

    def mouse_down(self, button: int, xpos: float, ypos: float) -> None:
        """Mouse button pressed."""
        pass

    def mouse_up(self, button: int, xpos: float, ypos: float) -> None:
        """Mouse button released."""
        pass

    def mouse_move(self, xpos: float, ypos: float) -> None:
        """Mouse moved."""
        pass

    def mouse_wheel(self, direction: int) -> None:
        """
        Mouse wheel scrolled.
        direction: +1 (up), -1 (down)
        """
        pass

    def key_down(
        self,
        key: int,
        mod_control: bool,
        mod_alt: bool,
        mod_shift: bool,
    ) -> None:
        """Key pressed."""
        pass

    def key_up(
        self,
        key: int,
        mod_control: bool,
        mod_alt: bool,
        mod_shift: bool,
    ) -> None:
        """Key released."""
        pass
