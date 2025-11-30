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
from graphics_color import GraphicsColor
from graphics_shape_2d_instance import GraphicsShape2DInstance
from paddle import Paddle
from ball import Ball
from pong_state import PongState
from pong_net import PongNet
from pong_number import PongNumber
import random

rng = random.Random()

class PongScene(GraphicsScene):

    paddle_inset: int = 34
    def __init__(self, graphics: GraphicsLibrary, pipeline: GraphicsPipeline, assets: AssetBundle) -> None:
        super().__init__(graphics, pipeline)
        self.assets = assets

        self.left_score = 0
        self.right_score = 0

        self.mouse_x = float(graphics.frame_buffer_width) / 2.0
        self.mouse_y = float(graphics.frame_buffer_height) / 2.0
        
        self.ball = Ball(x=0.0, y=0.0)
        self.reset_ball()

        self.left_paddle = Paddle(x=0.0, y=0.0)
        self.right_paddle = Paddle(x=0.0, y=0.0)
        self.reset_paddles()

        self.dead_timer = float(0.0)
        self.dead_time = float(2.0)

        self.calculate_bounds()
        self.state = PongState.Idle

        self.net = PongNet()
        self.number_left = PongNumber()
        self.number_right = PongNumber()
        
        self.reset_numbers()

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def wake(self) -> None:
        ...

    def load_prepare(self) -> None:
        pass

    def load(self) -> None:
        self.left_paddle.load(self.assets, self.graphics)
        self.right_paddle.load(self.assets, self.graphics)
        self.ball.load(self.assets, self.graphics)

    def load_complete(self) -> None:
        self.resize()

    def resize(self) -> None:
        graphics = self.graphics
        self.reset_paddles()
        self.calculate_bounds()
        self.net.rebuild(self.graphics)
        self.reset_numbers()
        self.rebuild_number_left(self.left_score)
        self.rebuild_number_right(self.right_score)
    
    # --------------------------------------------------------------
    # Main loop functions
    # --------------------------------------------------------------

    def update(self, dt: float) -> None:
        # No prints here
        self.left_paddle.update(dt=dt)
        self.right_paddle.update(dt=dt)
        self.ball.update(dt=dt)

        if self.state == PongState.Idle:
            self.update_idle(dt)
        elif self.state == PongState.Dead:
            self.update_dead(dt)
        elif self.state == PongState.Playing:
            self.update_play(dt)
        
    def draw(self) -> None:

        shape_program = self.pipeline.program_shape2d
        sprite_program = self.pipeline.program_sprite2d

        width = float(self.graphics.frame_buffer_width)
        height = float(self.graphics.frame_buffer_height)

        # No prints here
        projection_matrix = GraphicsMatrix()
        projection_matrix.ortho_size(width=width, height=height)

        self.graphics.clear_rgb(0.04, 0.04, 0.08)

        self.graphics.blend_set_alpha()

        self.number_left.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.number_right.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        
        self.left_paddle.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.right_paddle.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.ball.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)

        self.net.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    def mouse_down(self, button: int, xpos: float, ypos: float) -> None:
        self.mouse_x = xpos
        self.mouse_y = ypos
        if self.state == PongState.Idle:
            self.play()

    def mouse_up(self, button: int, xpos: float, ypos: float) -> None:
        self.mouse_x = xpos
        self.mouse_y = ypos
        

    def mouse_move(self, xpos: float, ypos: float) -> None:
        self.mouse_x = xpos
        self.mouse_y = ypos
        

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
        self.net.dispose()
        self.left_paddle.dispose()
        self.right_paddle.dispose()
        self.ball.dispose()

    def select_speed(self) -> float:
        width = float(self.graphics.frame_buffer_width)
        height = float(self.graphics.frame_buffer_height)
        return max(width, height) * rng.uniform(0.4, 0.6)

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

    def reset_numbers(self):
        graphics = self.graphics
        self.number_left.x = float(graphics.frame_buffer_width) / 4.0
        self.number_left.y = float(150.0)
        self.number_right.x = float(graphics.frame_buffer_width * 3) / 4.0
        self.number_right.y = float(150.0)

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
        self.ball.x_speed = self.select_speed()
        self.ball.y_speed = self.select_speed()
        if rng.choice([True, False]):
            self.ball.x_speed = -self.ball.x_speed
        if rng.choice([True, False]):
            self.ball.y_speed = -self.ball.y_speed
    
    def idle(self):
        self.state = PongState.Idle
        self.rebuild_number_left(0)
        self.rebuild_number_right(0)

    def dead(self):
        self.state = PongState.Dead
        self.dead_timer = 0.0

    def update_idle(self, dt: float) -> None:
        self.reset_ball()
        self.reset_paddles()
        self.left_score = 0
        self.right_score = 0

    def update_play(self, dt: float) -> None:
        
        self.ball.x += self.ball.x_speed * dt

        if self.ball.x <= self.left:
            self.ball.x = self.left
            self.ball.x_speed = self.select_speed()
            self.rebuild_number_left(self.left_score + random.randint(10, 100))
            
        if self.ball.x >= self.right:
            self.ball.x = self.right
            self.ball.x_speed = -self.select_speed()
            self.rebuild_number_right(self.right_score + random.randint(10, 100))
            

    def update_dead(self, dt: float) -> None:
        self.reset_ball()
        self.reset_paddles()
        self.dead_timer += dt
        if self.dead_timer >= self.dead_time:
            self.idle()

    def rebuild_number_left(self, new_score: int) -> None:
        self.left_score = new_score
        self.number_left.rebuild(new_score, self.graphics, self.assets)
    
    def rebuild_number_right(self, new_score: int) -> None:
        self.right_score = new_score
        self.number_right.rebuild(new_score, self.graphics, self.assets)

        
        


