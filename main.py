# main.py

import sys
import ctypes
from pathlib import Path
import glfw
from OpenGL import GL as gl
import time
from graphics.graphics_app_shell import GraphicsAppShell
from graphics.graphics_pipeline import GraphicsPipeline
from graphics.graphics_library import GraphicsLibrary
from pong.pong_asset_bundle import PongAssetBundle
from pong.pong_scene import PongScene

def framebuffer_size_callback(window, width, height):
    app_shell = glfw.get_window_user_pointer(window)
    if app_shell:
        screen_width, screen_height = glfw.get_window_size(window)
        screen_scale_x, screen_scale_y = glfw.get_window_content_scale(window)
        frame_buffer_width, frame_buffer_height = glfw.get_framebuffer_size(window)
        if app_shell.scene:
            scene = app_shell.scene
            if scene.graphics:
                graphics = scene.graphics
                graphics.resize(screen_width=screen_width,
                                screen_height=screen_height,
                                screen_scale_x=screen_scale_x,
                                screen_scale_y=screen_scale_y,
                                frame_buffer_width=frame_buffer_width,
                                frame_buffer_height=frame_buffer_height)
        app_shell.resize()

def key_callback(window, key, scancode, action, mods):

    # We treat CTRL and CMD/SUPER as equivalent
    # On windows, CTRL+C is copy, on mac SUPER+C is copy
    mod_control = bool(mods & (glfw.MOD_CONTROL | glfw.MOD_SUPER))
    mod_shift = bool(mods & glfw.MOD_SHIFT)
    mod_alt = bool(mods & glfw.MOD_ALT)

    # --- Forward to AppShell ---------------------------------------
    app_shell = glfw.get_window_user_pointer(window)
    if app_shell:
        if action == glfw.PRESS:
            app_shell.key_down(
                key=key,
                mod_control=mod_control,
                mod_alt=mod_alt,
                mod_shift=mod_shift,
            )
        elif action == glfw.RELEASE:
            app_shell.key_up(
                key=key,
                mod_control=mod_control,
                mod_alt=mod_alt,
                mod_shift=mod_shift,
            )

    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
def mouse_button_callback(window, button, action, mods):
    # Map GLFW button → our (-1, 0, 1)
    if button == glfw.MOUSE_BUTTON_LEFT:
        mapped_button = -1
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        mapped_button = 0
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        mapped_button = 1
    else:
        mapped_button = None  # unknown or extra buttons
    
    xpos, ypos = glfw.get_cursor_pos(window)
    app_shell = glfw.get_window_user_pointer(window)
    if app_shell and mapped_button is not None:
        if app_shell.scene:
            if app_shell.scene.graphics:
                xpos *= app_shell.scene.graphics.screen_scale_x
                ypos *= app_shell.scene.graphics.screen_scale_y
        if action == glfw.PRESS:
            app_shell.mouse_down(
                button=mapped_button,
                xpos=xpos,
                ypos=ypos,
            )
        elif action == glfw.RELEASE:
            app_shell.mouse_up(
                button=mapped_button,
                xpos=xpos,
                ypos=ypos,
            )

def cursor_pos_callback(window, xpos, ypos):
    app_shell = glfw.get_window_user_pointer(window)
    if app_shell:
        if app_shell.scene:
            if app_shell.scene.graphics:
                xpos *= app_shell.scene.graphics.screen_scale_x
                ypos *= app_shell.scene.graphics.screen_scale_y
        app_shell.mouse_move(xpos=xpos, ypos=ypos)

def scroll_callback(window, xoffset, yoffset):
    direction = 0
    if yoffset > 0:
        direction = -1
    elif yoffset < 0:
        direction = 1
    app_shell = glfw.get_window_user_pointer(window)
    if app_shell:
        app_shell.mouse_wheel(direction=direction)

def main():
    
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

    screen_width, screen_height = glfw.get_window_size(window)
    screen_scale_x, screen_scale_y = glfw.get_window_content_scale(window)
    frame_buffer_width, frame_buffer_height = glfw.get_framebuffer_size(window)
    
    pipeline = GraphicsPipeline()
    graphics = GraphicsLibrary(screen_width=screen_width,
                               screen_height=screen_height,
                               screen_scale_x=screen_scale_x,
                               screen_scale_y=screen_scale_y,
                               frame_buffer_width=frame_buffer_width,
                               frame_buffer_height=frame_buffer_height)
    
    assets = PongAssetBundle()
    pong_scene = PongScene(graphics=graphics, pipeline=pipeline, assets=assets)
    app_shell = GraphicsAppShell(scene=pong_scene)

    glfw.set_window_user_pointer(window, app_shell)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # The Keyboard
    glfw.set_key_callback(window, key_callback)

    # The Mouse
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    app_shell.wake()

    assets.load(graphics=graphics)
    app_shell.prepare()

    previous_time = time.time()
    while not glfw.window_should_close(window):
        current_time = time.time()
        dt = current_time - previous_time
        dt = min(dt, 0.1)
        previous_time = current_time
        app_shell.update(dt)
        app_shell.draw()
        glfw.swap_buffers(window)
        glfw.poll_events()

    app_shell.dispose()
    assets.dispose()
    glfw.terminate()

if __name__ == "__main__":
    main()
