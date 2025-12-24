# pong_scene.py
from __future__ import annotations
from OpenGL import GL as gl
from graphics.graphics_scene import GraphicsScene
from pong.pong_asset_bundle import PongAssetBundle
from graphics.graphics_primitives import Sprite2DVertex
from graphics.graphics_array_buffer import GraphicsArrayBuffer
from graphics.graphics_library import GraphicsLibrary
from graphics.graphics_pipeline import GraphicsPipeline
from graphics.graphics_matrix import GraphicsMatrix
from graphics.graphics_color import GraphicsColor
from graphics.graphics_shape_2d_instance import GraphicsShape2DInstance
from pong.pong_paddle import PongPaddle
from pong.pong_ball import PongBall
from pong.pong_state import PongState
from pong.pong_net import PongNet
from pong.pong_number import PongNumber
import random

rng = random.Random()

class PongScene(GraphicsScene):

    paddle_inset: int = 34
    def __init__(self,
                 graphics: GraphicsLibrary,
                 pipeline: GraphicsPipeline,
                 assets: PongAssetBundle) -> None:
        
        super().__init__(graphics, pipeline)

        self.assets = assets

        self.left_score = 0
        self.right_score = 0

        self.mouse_x = float(graphics.frame_buffer_width) / 2.0
        self.mouse_y = float(graphics.frame_buffer_height) / 2.0
        
        self.ball = PongBall(x=0.0, y=0.0)
        self.reset_ball()

        self.left_paddle = PongPaddle(x=0.0, y=0.0)
        self.right_paddle = PongPaddle(x=0.0, y=0.0)
        self.reset_paddles()

        self.dead_timer = float(0.0)
        self.dead_time = float(4.0)

        self.calculate_bounds()
        self.state = PongState.Idle

        self.net = PongNet()
        self.number_left = PongNumber()
        self.number_right = PongNumber()
        
        self.reset_numbers()
    
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
        shape_program = self.pipeline.program_shape_2d
        sprite_program = self.pipeline.program_sprite_2d
        width = float(self.graphics.frame_buffer_width)
        height = float(self.graphics.frame_buffer_height)
        projection_matrix = GraphicsMatrix()
        projection_matrix.ortho_size(width=width, height=height)
        self.graphics.clear_rgb(0.04, 0.04, 0.08)
        self.graphics.blend_set_disabled()
        self.net.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.graphics.blend_set_alpha()
        self.number_left.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.number_right.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.left_paddle.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.right_paddle.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)
        self.ball.draw(graphics=self.graphics, pipeline=self.pipeline, projection_matrix=projection_matrix)

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
        paddle_height_2 = paddle_height / 2.0
        paddle_inset = float(PongScene.paddle_inset)
        self.top = ball_height_2
        self.bottom = height - ball_height_2
        self.left = ball_width_2
        self.right = width - ball_width_2
        self.paddle_top = paddle_height_2
        self.paddle_bottom = height - paddle_height_2
        self.paddle_left = paddle_inset + paddle_width_2 + ball_width_2
        self.paddle_right = width - paddle_inset - paddle_width_2 - ball_width_2

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
        self.left_paddle.is_red = False
        self.right_paddle.is_red = False
        self.ball.is_red = False

    def dead(self):
        self.state = PongState.Dead
        self.dead_timer = 0.0
        self.left_paddle.is_red = True
        self.right_paddle.is_red = True
        self.ball.is_red = True

    def update_idle(self, dt: float) -> None:
        self.reset_ball()
        self.reset_paddles()
        self.left_score = 0
        self.right_score = 0

    def update_play(self, dt: float) -> None:

        width = float(self.graphics.frame_buffer_width)
        height = float(self.graphics.frame_buffer_height)

        paddle_track_strength = max(width, height) * 0.25

        paddle_factor_multiply = float(4.0)
        paddle_factor_add = float(200.0)
        
        if self.left_paddle.y < self.mouse_y:
            amount = (self.mouse_y - self.left_paddle.y) * dt * paddle_factor_multiply + paddle_factor_add * dt
            self.left_paddle.y += amount
            if self.left_paddle.y > self.mouse_y:
                self.left_paddle.y = self.mouse_y

        elif self.left_paddle.y > self.mouse_y:
            amount = (self.left_paddle.y - self.mouse_y) * dt * paddle_factor_multiply + paddle_factor_add * dt
            self.left_paddle.y -= amount
            if self.left_paddle.y < self.mouse_y:
                self.left_paddle.y = self.mouse_y

        if self.left_paddle.y > self.paddle_bottom:
            self.left_paddle.y = self.paddle_bottom
        
        if self.left_paddle.y < self.paddle_top:
            self.left_paddle.y = self.paddle_top

        if self.right_paddle.y < self.ball.y:
            amount = (self.ball.y - self.right_paddle.y) * dt * paddle_factor_multiply + paddle_factor_add * dt
            self.right_paddle.y += amount
            if self.right_paddle.y > self.ball.y:
                self.right_paddle.y = self.ball.y

        elif self.right_paddle.y > self.ball.y:
            amount = (self.right_paddle.y - self.ball.y) * dt * paddle_factor_multiply + paddle_factor_add * dt
            self.right_paddle.y -= amount
            if self.right_paddle.y < self.ball.y:
                self.right_paddle.y = self.ball.y

        if self.right_paddle.y > self.paddle_bottom:
            self.right_paddle.y = self.paddle_bottom
        
        if self.right_paddle.y < self.paddle_top:
            self.right_paddle.y = self.paddle_top
        
        self.ball.x += self.ball.x_speed * dt
        self.ball.y += self.ball.y_speed * dt
        
        if self.ball.y <= self.top:
            self.ball.y_speed = self.select_speed()
        if self.ball.y >= self.bottom:
            self.ball.y_speed = -self.select_speed()
        
        if self.ball.x_speed < 0.0 and self.vertical_overlap(self.left_paddle):
            if self.ball.x <= self.paddle_left:
                self.ball.x_speed = self.select_speed()
                self.left_score += 1
                self.rebuild_number_left(self.left_score)
                
        if self.ball.x_speed > 0.0 and self.vertical_overlap(self.right_paddle):
            if self.ball.x >= self.paddle_right:
                self.ball.x_speed = -self.select_speed()
                self.right_score += 1
                self.rebuild_number_right(self.right_score)

        if self.ball.x <= self.left:
            self.dead()
            return
        
        if self.ball.x >= self.right:
            self.dead()
            return

    def vertical_overlap(self, paddle: Paddle) -> bool:
        top = paddle.y - (paddle.height / 2.0) - (self.ball.height / 2.0)
        bottom = paddle.y + (paddle.height / 2.0) + (self.ball.height / 2.0)
        if self.ball.y >= top and self.ball.y <= bottom:
            return True
        else:
            return False
        
    def update_dead(self, dt: float) -> None:
        self.dead_timer += dt
        if self.dead_timer >= self.dead_time:
            self.idle()

    def rebuild_number_left(self, new_score: int) -> None:
        self.number_left.rebuild(new_score, self.graphics, self.assets)
    
    def rebuild_number_right(self, new_score: int) -> None:
        self.number_right.rebuild(new_score, self.graphics, self.assets)

        
        


