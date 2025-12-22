# graphics_pipeline.py

import os
from OpenGL.GL import (
    glCreateShader,
    glShaderSource,
    glCompileShader,
    glGetShaderiv,
    glGetShaderInfoLog,
    glDeleteShader,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS,
)

from graphics.shader_program_sprite_2d import ShaderProgramSprite2D
from graphics.shader_program_shape_2d import ShaderProgramShape2D
from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils
from typing import Optional

class GraphicsPipeline:
    def __init__(self):
        
        # Sprite 2D shader functions and program
        self.function_sprite2d_vertex = self._load_shader_vertex("graphics/shaders/", "sprite_2d_vertex", "glsl")
        self.function_sprite2d_fragment = self._load_shader_fragment("graphics/shaders/", "sprite_2d_fragment", "glsl")
        self.program_sprite_2d = ShaderProgramSprite2D(
            "sprite_2d",
            self.function_sprite2d_vertex,
            self.function_sprite2d_fragment,
        )

        # Shape 2D shader functions and program
        self.function_shape2d_vertex = self._load_shader_vertex("graphics/shaders/", "shape_2d_vertex", "glsl")
        self.function_shape2d_fragment = self._load_shader_fragment("graphics/shaders/", "shape_2d_fragment", "glsl")
        self.program_shape_2d = ShaderProgramShape2D(
            "shape_2d",
            self.function_shape2d_vertex,
            self.function_shape2d_fragment,
        )

    # ---------------------------------------------------------
    # Shader loading helpers
    # ---------------------------------------------------------

    def _load_shader_vertex(self, subdirectory: Optional[str], name: str, extension: str) -> int:
        return self._load_shader(GL_VERTEX_SHADER, subdirectory, name, extension)

    def _load_shader_fragment(self, subdirectory: Optional[str], name: str, extension: str) -> int:
        return self._load_shader(GL_FRAGMENT_SHADER, subdirectory, name, extension)
    
    def _load_shader(self, shader_type: int, subdirectory: Optional[str], name: str, extension: str) -> int:
        try:
            source = FileUtils.load_local_text(subdirectory, name, extension)
        except OSError as e:
            print(f"[GraphicsPipeline] Failed to read shader '{name}': {e}")
            return 0

        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if glGetShaderiv(shader, GL_COMPILE_STATUS) == 0:
            log = glGetShaderInfoLog(shader)
            print(f"[ShaderCompile] Error compiling '{name}': {log}")
            glDeleteShader(shader)
            return 0
        return shader
