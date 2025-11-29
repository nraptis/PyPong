# main.py
import sys
import ctypes
from pathlib import Path
from random import random

import glfw
import numpy as np
from OpenGL import GL as gl
from PIL import Image

from graphics_matrix import GraphicsMatrix

from graphics_pipeline import GraphicsPipeline
from graphics_library import GraphicsLibrary

from primitives import Sprite2DVertex
from primitives import Shape2DVertex

from graphics_array_buffer import GraphicsArrayBuffer
from graphics_texture import GraphicsTexture
from graphics_sprite import GraphicsSprite

def framebuffer_size_callback(window, width, height):
    # Update OpenGL viewport
    gl.glViewport(0, 0, width, height)

    # Update your GraphicsLibrary dimensions
    graphics = glfw.get_window_user_pointer(window)
    graphics.width = width
    graphics.height = height

    print("Resized:", width, height)

def key_callback(window, key, scancode, action, mods):

    print("action = ", action)
    print("mods = ", mods)

    mod_shift = False
    if mods & glfw.MOD_SHIFT:
        mod_shift = True

    # We will count CTRL and CMD as same thing...
    mod_control = False
    if mods & glfw.MOD_CONTROL or mods & glfw.MOD_SUPER:
        mod_control = True

    mod_alt = False
    if mods & glfw.MOD_ALT:
        mod_alt = True

    print("ctrl:", mod_control, "alt:", mod_alt, "shift:", mod_shift)
    
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    step = 10.0
    if key == glfw.KEY_LEFT:
        print("left")
    if key == glfw.KEY_RIGHT:
        print("right")
    if key == glfw.KEY_UP:
        print("up")
    if key == glfw.KEY_DOWN:
        print("down")

    zoom_step = 0.1
    if key == glfw.KEY_Z:
        print("z")
    if key == glfw.KEY_X:
        print("x")


def mouse_button_callback(window, button, action, mods):

    if button == glfw.MOUSE_BUTTON_LEFT:
        print("Left,", action)
        
    if button == glfw.MOUSE_BUTTON_RIGHT:
        print("Right,", action)
        
    if button == glfw.MOUSE_BUTTON_MIDDLE:
        print("Middle,", action)
    
def cursor_pos_callback(window, xpos, ypos):
    pass

def scroll_callback(window, xoffset, yoffset):
    scroll_dir = 0
    if yoffset > 0:
        scroll_dir = -1
    elif yoffset < 0:
        scroll_dir = 1
    print("Scroll Dir: ", scroll_dir)

# ----------------------------------------------------------------------
# Main: Textured triangle using vanilla OpenGL + your sprite_2d shaders
# ----------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # Initialize GLFW and create a window / context
    # --------------------------------------------------------------
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    width, height = 1280, 960
    window = glfw.create_window(width, height, "PyPong (OpenGL)", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        sys.exit(1)
        
    print("GL VERSION:", gl.glGetString(gl.GL_VERSION))
    print("GLSL VERSION:", gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION))
    print("VENDOR:", gl.glGetString(gl.GL_VENDOR))
    
    glfw.make_context_current(window)

    base_dir = Path(__file__).resolve().parent
    shader_path = base_dir / "shaders"
    image_path = base_dir / "images/digit_9.png"

    pipeline = GraphicsPipeline(shader_path)
    graphics = GraphicsLibrary(width=width, height=height)

    glfw.set_window_user_pointer(window, graphics)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # The Keyboard
    glfw.set_key_callback(window, key_callback)

    # The Mouse
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    
    texture = GraphicsTexture(graphics=graphics, file_name=image_path)
    texture.print()

    sprite = GraphicsSprite()
    sprite.load(graphics=graphics, texture=texture)
    sprite.print()

    sprite_prog = pipeline.program_sprite2d

    sprite_vertices = [
        Sprite2DVertex(x=-140.0, y=-133.0, u=0.0, v=0.0),
        Sprite2DVertex(x=140.0,  y=-133.0, u=1.0, v=0.0),
        Sprite2DVertex(x=-140.0,  y=140.0, u=0.0, v=1.0),
        Sprite2DVertex(x=140.0,  y=140.0, u=1.0, v=1.0),
    ]

    sprite_vertex_buffer = GraphicsArrayBuffer[Sprite2DVertex]()
    sprite_vertex_buffer.load(graphics, sprite_vertices)

    sprite_indices = [0, 1, 2, 3]
    sprite_index_buffer = graphics.buffer_index_generate_from_int_array(sprite_indices)
    shape_prog = pipeline.program_shape2d

    shape_vertices = [
        Shape2DVertex(x=-128.0, y=-128.0),
        Shape2DVertex(x=128.0,  y=-128.0),
        Shape2DVertex(x=-128.0,  y=128.0),
        Shape2DVertex(x=128.0,  y=128.0),
    ]

    shape_vertex_buffer = GraphicsArrayBuffer[Shape2DVertex]()
    shape_vertex_buffer.load(graphics, shape_vertices)

    shape_indices = [0, 1, 2, 3]
    shape_index_buffer = graphics.buffer_index_generate_from_int_array(shape_indices)

    roz = float(0.0)

    while not glfw.window_should_close(window):

        roz += 0.5

        projection = GraphicsMatrix()
        projection.ortho_size(width=graphics.width, height=graphics.height)

        model_view = GraphicsMatrix()
        model_view.translate(x=width/2, y=height/2, z=0.0)
        #model_view.rotate_z(roz * 0.04)
        #model_view.scale(2.0)

        graphics.clear_rgb(0.22, 0.22, 0.28)

        graphics.blend_set_alpha()

        graphics.link_buffer_to_shader_program_array_buffer(sprite_prog, sprite_vertex_buffer)
        graphics.uniforms_texture_set_sprite(program=sprite_prog, sprite=sprite)
        graphics.uniforms_modulate_color_set(sprite_prog, r=1.0, g=1.0, b=0.5, a=0.5)
        graphics.uniforms_matrices_set(shape_prog, projection, model_view)
        graphics.draw_primitives(index_buffer=sprite_index_buffer, primitive_type=gl.GL_TRIANGLE_STRIP, count=4)
        graphics.unlink_buffer_from_shader_program(sprite_prog)

        graphics.link_buffer_to_shader_program_array_buffer(shape_prog, shape_vertex_buffer)
        graphics.uniforms_matrices_set(shape_prog, projection, model_view)
        graphics.uniforms_modulate_color_set(shape_prog, r=1.0, g=0.25, b=0.5, a=0.5)
        graphics.draw_primitives(index_buffer=shape_index_buffer, primitive_type=gl.GL_TRIANGLE_STRIP, count=4)
        graphics.unlink_buffer_from_shader_program(shape_prog)

        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    #gl.glDeleteTextures([tex])

    glfw.terminate()

if __name__ == "__main__":
    main()
