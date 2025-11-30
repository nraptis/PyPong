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
from paddle import Paddle
from ball import Ball
from pong_state import PongState

class PongScene(GraphicsScene):

    paddle_inset: int = 34
    def __init__(self, graphics, pipeline, assets: AssetBundle) -> None:
        super().__init__(graphics, pipeline)
        self.assets = assets

        self.left_score = 33
        self.right_score = 47

        self.mouse_x = float(graphics.frame_buffer_width) / 2.0
        self.mouse_y = float(graphics.frame_buffer_height) / 2.0
        
        self.ball = Ball(x=0.0, y=0.0)
        self.reset_ball()

        self.left_paddle = Paddle(x=0.0, y=0.0)
        self.right_paddle = Paddle(x=0.0, y=0.0)
        self.reset_paddles()

        self.state = PongState.Idle

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def wake(self) -> None:
        print("PongScene.wake()")

    def load_prepare(self) -> None:
        pass

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

        self.left_paddle.load(self.assets, self.graphics)
        self.right_paddle.load(self.assets, self.graphics)
        self.ball.load(self.assets, self.graphics)

    def load_complete(self) -> None:
        pass

    def resize(self) -> None:
        graphics = self.graphics
        self.reset_paddles()
    
    # --------------------------------------------------------------
    # Main loop functions
    # --------------------------------------------------------------

    def update(self, dt: float) -> None:
        # No prints here
        self.left_paddle.update(dt=dt)
        self.right_paddle.update(dt=dt)
        self.ball.update(dt=dt)
        self.ball.x += dt * 10.0

    def draw(self) -> None:

        sprite_prog = self.pipeline.program_sprite2d

        width = float(self.graphics.frame_buffer_width)
        height = float(self.graphics.frame_buffer_height)
        

        # No prints here
        projection_matrix = GraphicsMatrix()
        projection_matrix.ortho_size(width=width, height=height)

        model_view = GraphicsMatrix()
        model_view.translate(x=width/2, y=height/2, z=0.0)
        #model_view.rotate_z(roz * 0.04)
        #model_view.scale(2.0)

        self.graphics.clear_rgb(0.22, 0.22, 0.28)

        self.graphics.blend_set_alpha()

        self.graphics.link_buffer_to_shader_program(sprite_prog, self.sprite_vertex_buffer)
        self.graphics.uniforms_texture_set_sprite(program=sprite_prog, sprite=self.assets.ball_sprite)
        self.graphics.uniforms_modulate_color_set(sprite_prog, r=1.0, g=1.0, b=0.5, a=0.5)
        self.graphics.uniforms_matrices_set(sprite_prog, projection_matrix, model_view)
        self.graphics.draw_primitives(index_buffer=self.sprite_index_buffer, primitive_type=gl.GL_TRIANGLE_STRIP, count=4)
        self.graphics.unlink_buffer_from_shader_program(sprite_prog)
        
        self.left_paddle.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.right_paddle.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.ball.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)

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
       if self.ball:
           self.ball.x = xpos
           self.ball.y = ypos

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

    def calculate_bounds(self):

        width = float(self.graphics.frame_buffer_width)
        height = float(self.graphics.frame_buffer_height)

        ball_width = float(self.assets.ball_width)
        ball_width_2 = ball_width / 2.0
        ball_height = float(self.assets.ball_height)
        ball_height_2 = ball_height / 2.0
        
        paddle_width = float(self.assets.paddle_width)
        paddle_width_2 = paddle_width / 2.0
        paddle_height = float(self.assets.paddle_height)

        self.top = ball_height_2
        self.bottom = height - ball_height_2
        self.left = ball_width_2
        self.right = width - ball_width_2



    def reset_ball(self):
        graphics = self.graphics
        self.ball.x = float(graphics.frame_buffer_width) / 2.0
        self.ball.y = float(graphics.frame_buffer_height) / 2.0
    
    def reset_paddles(self):
        graphics = self.graphics
        self.left_paddle.x = 0.0 + float(PongScene.paddle_inset)
        self.left_paddle.y = float(graphics.frame_buffer_height) / 2.0
        self.right_paddle.x = graphics.frame_buffer_width - float(PongScene.paddle_inset)
        self.right_paddle.y = float(graphics.frame_buffer_height) / 2.0

    def play(self):
        self.state = PongState.Playing



