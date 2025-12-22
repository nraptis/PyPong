# app_shell.py
from __future__ import annotations
from graphics.graphics_scene import GraphicsScene

class GraphicsAppShell:
    def __init__(self, scene: GraphicsScene) -> None:
        self.scene = scene

    # -------- Lifecycle -------------------------------------------------

    def wake(self) -> None:
        self.scene.wake()

    def prepare(self) -> None:
        self.scene.load_prepare()
        self.scene.load()
        self.scene.load_complete()

    def resize(self) -> None:
        self.scene.resize()

    def update(self, dt: float) -> None:
        self.scene.update(dt=dt)

    def draw(self) -> None:
        self.scene.draw()

    def dispose(self) -> None:
        self.scene.dispose()

    # -------- Input -----------------------------------------------------

    def mouse_down(self, button: int, xpos: float, ypos: float) -> None:
        self.scene.mouse_down(button=button, xpos=xpos, ypos=ypos)

    def mouse_up(self, button: int, xpos: float, ypos: float) -> None:
        self.scene.mouse_up(button=button, xpos=xpos, ypos=ypos)

    def mouse_move(self, xpos: float, ypos: float) -> None:
        self.scene.mouse_move(xpos=xpos, ypos=ypos)

    def mouse_wheel(self, direction: int) -> None:
        self.scene.mouse_wheel(direction=direction)

    def key_down(
        self,
        key: int,
        mod_control: bool,
        mod_alt: bool,
        mod_shift: bool,
    ) -> None:
        self.scene.key_down(
            key=key,
            mod_control=mod_control,
            mod_alt=mod_alt,
            mod_shift=mod_shift,
        )

    def key_up(
        self,
        key: int,
        mod_control: bool,
        mod_alt: bool,
        mod_shift: bool,
    ) -> None:
        self.scene.key_up(
            key=key,
            mod_control=mod_control,
            mod_alt=mod_alt,
            mod_shift=mod_shift,
        )
