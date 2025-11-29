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

from shader_program_sprite_2d import ShaderProgramSprite2D
from shader_program_shape_2d import ShaderProgramShape2D

class GraphicsPipeline:
    def __init__(self, base_path: str = "."):
        self.base_path = base_path

        # Sprite 2D shader functions and program
        self.function_sprite2d_vertex = self._load_shader_vertex("sprite_2d_vertex.glsl")
        self.function_sprite2d_fragment = self._load_shader_fragment("sprite_2d_fragment.glsl")
        self.program_sprite2d = ShaderProgramSprite2D(
            "sprite_2d",
            self.function_sprite2d_vertex,
            self.function_sprite2d_fragment,
        )

        # Shape 2D shader functions and program
        self.function_shape2d_vertex = self._load_shader_vertex("shape_2d_vertex.glsl")
        self.function_shape2d_fragment = self._load_shader_fragment("shape_2d_fragment.glsl")
        self.program_shape2d = ShaderProgramShape2D(
            "shape_2d",
            self.function_shape2d_vertex,
            self.function_shape2d_fragment,
        )

    # ---------------------------------------------------------
    # Shader loading helpers
    # ---------------------------------------------------------

    def _read_file_as_string(self, filename: str) -> str:
        full_path = os.path.join(self.base_path, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_shader_vertex(self, filename: str) -> int:
        return self._load_shader(GL_VERTEX_SHADER, filename)

    def _load_shader_fragment(self, filename: str) -> int:
        return self._load_shader(GL_FRAGMENT_SHADER, filename)

    def _load_shader(self, shader_type: int, filename: str) -> int:
        try:
            source = self._read_file_as_string(filename)
        except OSError as e:
            print(f"[GraphicsPipeline] Failed to read shader '{filename}': {e}")
            return 0

        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if glGetShaderiv(shader, GL_COMPILE_STATUS) == 0:
            log = glGetShaderInfoLog(shader)
            print(f"[ShaderCompile] Error compiling '{filename}': {log}")
            glDeleteShader(shader)
            return 0
        return shader
