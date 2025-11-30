# pong_scene.py
from __future__ import annotations
from OpenGL import GL as gl
from graphics_scene import GraphicsScene
from asset_bundle import AssetBundle
from primitives import Sprite2DVertex
from graphics_array_buffer import GraphicsArrayBuffer
from graphics_library import GraphicsLibrary
from graphics_pipeline import GraphicsPipeline
from graphics_matrix import GraphicsMatrix

class PongScene(GraphicsScene):
    """
    A skeleton Pong scene.
    Prints function calls (except update, draw, mouse_move).
    """

    def __init__(self, graphics, pipeline, assets: AssetBundle) -> None:
        super().__init__(graphics, pipeline)
        self.assets = assets

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def wake(self) -> None:
        print("PongScene.wake()")

    def load_prepare(self) -> None:
        print("PongScene.load_prepare()")

    def load(self) -> None:
        print("PongScene.load()")

        self.sprite_vertices = [
            Sprite2DVertex(x=-140.0, y=-133.0, u=0.0, v=0.0),
            Sprite2DVertex(x=140.0,  y=-133.0, u=1.0, v=0.0),
            Sprite2DVertex(x=-140.0,  y=140.0, u=0.0, v=1.0),
            Sprite2DVertex(x=140.0,  y=140.0, u=1.0, v=1.0),
        ]

        self.sprite_vertex_buffer = GraphicsArrayBuffer[Sprite2DVertex]()
        self.sprite_vertex_buffer.load(self.graphics, self.sprite_vertices)

        self.sprite_indices = [0, 1, 2, 3]
        self.sprite_index_buffer = self.graphics.buffer_index_generate_from_int_array(self.sprite_indices)

    def load_complete(self) -> None:
        print("PongScene.load_complete()")

    def resize(self, width: int, height: int) -> None:
        print(f"PongScene.resize(width={width}, height={height})")

    # --------------------------------------------------------------
    # Main loop functions (NO PRINTS)
    # --------------------------------------------------------------

    def update(self, dt: float) -> None:
        # No prints here
        pass

    def draw(self) -> None:

        sprite_prog = self.pipeline.program_sprite2d

        width = float(self.graphics.width)
        height = float(self.graphics.height)
        

        # No prints here
        projection = GraphicsMatrix()
        projection.ortho_size(width=width, height=height)

        model_view = GraphicsMatrix()
        model_view.translate(x=width/2, y=height/2, z=0.0)
        #model_view.rotate_z(roz * 0.04)
        #model_view.scale(2.0)

        self.graphics.clear_rgb(0.22, 0.22, 0.28)

        self.graphics.blend_set_alpha()

        self.graphics.link_buffer_to_shader_program_array_buffer(sprite_prog, self.sprite_vertex_buffer)
        self.graphics.uniforms_texture_set_sprite(program=sprite_prog, sprite=self.assets.ball_sprite)
        self.graphics.uniforms_modulate_color_set(sprite_prog, r=1.0, g=1.0, b=0.5, a=0.5)
        self.graphics.uniforms_matrices_set(sprite_prog, projection, model_view)
        self.graphics.draw_primitives(index_buffer=self.sprite_index_buffer, primitive_type=gl.GL_TRIANGLE_STRIP, count=4)
        self.graphics.unlink_buffer_from_shader_program(sprite_prog)


    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    def mouse_down(self, button: int, xpos: float, ypos: float) -> None:
        print(
            f"PongScene.mouse_down(button={button}, xpos={xpos}, ypos={ypos})"
        )

    def mouse_up(self, button: int, xpos: float, ypos: float) -> None:
        print(
            f"PongScene.mouse_up(button={button}, xpos={xpos}, ypos={ypos})"
        )

    def mouse_move(self, xpos: float, ypos: float) -> None:
        # No prints here
        pass

    def mouse_wheel(self, direction: int) -> None:
        print(f"PongScene.mouse_wheel(direction={direction})")

    def key_down(
        self,
        key: int,
        mod_control: bool,
        mod_alt: bool,
        mod_shift: bool,
    ) -> None:
        print(
            f"PongScene.key_down("
            f"key={key}, "
            f"mod_control={mod_control}, "
            f"mod_alt={mod_alt}, "
            f"mod_shift={mod_shift}"
            f")"
        )

    def key_up(
        self,
        key: int,
        mod_control: bool,
        mod_alt: bool,
        mod_shift: bool,
    ) -> None:
        print(
            f"PongScene.key_up("
            f"key={key}, "
            f"mod_control={mod_control}, "
            f"mod_alt={mod_alt}, "
            f"mod_shift={mod_shift}"
            f")"
        )

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    def dispose(self) -> None:
        print("PongScene.dispose()")
