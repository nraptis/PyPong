# graphics_library.py

from __future__ import annotations
from typing import Optional, Sequence, TypeVar
import numpy as np
from OpenGL import GL as gl
from graphics.graphics_float_bufferable import GraphicsFloatBufferable
from graphics.graphics_array_buffer import GraphicsArrayBuffer
from graphics.graphics_texture import GraphicsTexture
from graphics.graphics_sprite import GraphicsSprite
from graphics.graphics_color import GraphicsColor
from graphics.graphics_matrix import GraphicsMatrix
from graphics.shader_program import ShaderProgram

T = TypeVar("T", bound=GraphicsFloatBufferable)

class GraphicsLibrary:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        screen_scale_x: int,
        screen_scale_y: int,
        frame_buffer_width: int,
        frame_buffer_height: int
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_scale_x = screen_scale_x
        self.screen_scale_y = screen_scale_y
        self.frame_buffer_width = frame_buffer_width
        self.frame_buffer_height = frame_buffer_height
        gl.glViewport(0, 0, frame_buffer_width, frame_buffer_height)
        self.texture_set_filter_linear()
        self.texture_set_clamp()

    def resize(self,
        screen_width: int,
        screen_height: int,
        screen_scale_x: int,
        screen_scale_y: int,
        frame_buffer_width: int,
        frame_buffer_height: int) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_scale_x = screen_scale_x
        self.screen_scale_y = screen_scale_y
        self.frame_buffer_width = frame_buffer_width
        self.frame_buffer_height = frame_buffer_height
        gl.glViewport(0, 0, frame_buffer_width, frame_buffer_height)

    def clear(self) -> None:
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
    def clear_color(self, color: Optional[GraphicsColor]) -> None:
        if color:
            gl.glClearColor(color.r, color.g, color.b, 1.0)
        else:
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    def clear_rgb(self, r: float, g: float, b: float) -> None:
        gl.glClearColor(r, g, b, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)  
        
    # ----------------------------------------------------------------------
    # VBO helpers (ARRAY_BUFFER)
    # ----------------------------------------------------------------------

    def buffer_array_generate(self) -> int:
        buf_id = gl.glGenBuffers(1)
        if isinstance(buf_id, (list, tuple)):

            print("Generating Array Buffer @", buf_id[0])
            return int(buf_id[0])
        
        print("Generating Array Buffer @", buf_id)
        return int(buf_id)

    def buffer_array_delete(self, index: int | None) -> None:
        if index is None:
            return

        # --- Robust conversion to Python int ---
        try:
            # Handles: np.array([23]), np.uint32(23), plain int, etc.
            idx = int(index)
        except Exception:
            # If index is something unexpected: treat as "no buffer"
            return
        
        # Now safe to check
        if idx == -1:
            return

        print("Deleting Array Buffer @", idx)
        gl.glDeleteBuffers(1, [idx])

    def buffer_array_write(self, index: int | None, data: Sequence[float]) -> None:
        if index is None or index == -1:
            return
        arr = np.asarray(data, dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, index)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, arr, gl.GL_STATIC_DRAW)


    def buffer_array_bind(self, index: int | None) -> None:
        if index is None or index == -1:
            return
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, index)

    def buffer_array_bind_array_buffer(
        self,
        array_buffer: Optional[GraphicsArrayBuffer[T]],
    ) -> None:
        if array_buffer is None:
            return
        self.buffer_array_bind(index=array_buffer.buffer_index)

    # ----------------------------------------------------------------------
    # Index buffers (client-side numpy arrays for glDrawElements)
    # ----------------------------------------------------------------------
    def buffer_index_generate_from_list(self, values: Sequence[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.uint32)

    def buffer_index_write_from_list(
        self,
        values: Sequence[int],
        index_buffer: np.ndarray,
        count: Optional[int] = None,
    ) -> None:
        if count is None:
            count = len(values)
        count = min(count, len(values), index_buffer.size)
        index_buffer[:count] = np.asarray(values[:count], dtype=np.uint32)

    def buffer_index_delete(self, index: int | None) -> None:
        """
        Index buffers are client-side NumPy arrays, not GL buffers.
        Deletion is a no-op.
        """
        return
    
    # ----------------------------------------------------------------------
    # Float buffers (for GraphicsFloatBufferable -> list[float])
    # ----------------------------------------------------------------------
    def buffer_float_size(self, items: Sequence[T]) -> int:
        if not items:
            return 0
        element_size = items[0].size()
        return len(items) * element_size

    def buffer_float_generate_from_item(self, item: T) -> list[float]:
        buf: list[float] = []
        item.write_to_buffer(buf)
        return buf

    def buffer_float_generate_from_array(self, items: Sequence[T]) -> list[float]:
        buf: list[float] = []
        self.buffer_float_write_from_list(items, buf)
        return buf

    def buffer_float_write_from_list(
        self,
        items: Sequence[T],
        float_buffer: list[float],
        count: Optional[int] = None,
    ) -> None:
        float_buffer.clear()
        if count is None:
            count = len(items)
        limit = min(count, len(items))

        for i in range(limit):
            items[i].write_to_buffer(float_buffer)

    def buffer_float_write_from_item(self, item: T, float_buffer: list[float]) -> None:
        float_buffer.clear()
        item.write_to_buffer(float_buffer)

    # ----------------------------------------------------------------------
    # Texture state & creation
    # ----------------------------------------------------------------------

    def texture_set_filter_mipmap(self) -> None:
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)

    def texture_set_filter_linear(self) -> None:
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def texture_set_wrap_repeat(self) -> None:
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

    def texture_set_clamp(self) -> None:
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    def texture_bind_index(self, texture_index: int | None) -> None:
        """
        Bind a texture by index. Treat None and -1 as invalid.
        """
        if texture_index is None or texture_index == -1:
            return
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(texture_index))

    def texture_bind(self, texture: Optional[GraphicsTexture]) -> None:
        """
        Bind a texture object.
        """
        if texture is not None:
            self.texture_bind_index(texture.texture_index)

    def texture_delete(self, texture_index: int | None) -> None:
        """
        Delete a texture if the index is valid.
        Accepts None, Python ints, NumPy ints, or 1-element NumPy arrays.
        Treat None and -1 as invalid.
        Safe to call multiple times.
        """
        if texture_index is None:
            return
        
        try:
            idx = int(texture_index)
        except Exception:
            return

        if idx == -1:
            return

        print("Deleting Texture @", idx)
        gl.glDeleteTextures(1, [idx])
            
    # --- variant that takes a "bitmap" (you decide what that is) ----------
    def texture_generate_from_bitmap(self, bitmap) -> int:
        if bitmap is None:
            return -1

        # Expecting numpy array (H, W, 4), dtype=uint8
        data = np.asarray(bitmap, dtype=np.uint8)
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError("texture_generate_from_bitmap expects shape (H, W, 4) RGBA array")

        height, width, _ = data.shape

        tex = gl.glGenTextures(1)
        tex_id = int(tex[0] if isinstance(tex, (list, tuple)) else tex)
        if tex_id == 0:
            return -1

        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        self.texture_set_filter_linear()
        self.texture_set_clamp()

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            data,
        )

        print("Generating Texture @", tex_id)
        return tex_id

    # --- variant that creates a random RGBA texture -----------------------

    def texture_generate_random(self, width: int, height: int) -> int:
        width = int(width)
        height = int(height)

        tex = gl.glGenTextures(1)
        tex_id = int(tex[0] if isinstance(tex, (list, tuple)) else tex)
        if tex_id == 0:
            return -1

        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        self.texture_set_filter_linear()
        self.texture_set_clamp()

        # random RGBA, each channel 0..255
        pixels = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            pixels,
        )
        return tex_id

    # ----------------------------------------------------------------------
    # Blending
    # ----------------------------------------------------------------------

    def blend_set_alpha(self) -> None:
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def blend_set_additive(self) -> None:
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

    def blend_set_disabled(self) -> None:
        gl.glDisable(gl.GL_BLEND)

    # ----------------------------------------------------------------------
    # Draw helpers
    # ----------------------------------------------------------------------

    def draw_triangles(self, index_buffer: np.ndarray, count: int) -> None:
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            int(count),
            gl.GL_UNSIGNED_INT,
            index_buffer,
        )

    def draw_triangle_strips(self, index_buffer: Optional[np.ndarray], count: int) -> None:
        if index_buffer is None:
            return
        gl.glDrawElements(
            gl.GL_TRIANGLE_STRIP,
            int(count),
            gl.GL_UNSIGNED_INT,
            index_buffer,
        )

    def draw_primitives(self, index_buffer: np.ndarray, primitive_type: int, count: int) -> None:
        gl.glDrawElements(
            int(primitive_type),
            int(count),
            gl.GL_UNSIGNED_INT,
            index_buffer,
        )

    # ----------------------------------------------------------------------
    # Linking buffers to shader program (vertex attribs)
    # ----------------------------------------------------------------------
    def link_buffer_to_shader_program(
        self,
        program: Optional[ShaderProgram],
        buffer: Optional[GraphicsArrayBuffer[T]],
    ) -> None:
        """
        High-level helper: take a GraphicsArrayBuffer and link its VBO to the program.
        """
        if buffer is None:
            return
        self.link_buffer_to_shader_program_index(program, buffer.buffer_index)


    def link_buffer_to_shader_program_index(
        self,
        program: Optional[ShaderProgram],
        buffer_index: int | None,
    ) -> None:
        """
        Low-level helper: bind and configure a VBO by index.
        Treat None and -1 as invalid.
        """
        if program is None:
            return
        if program.program == 0:
            return
        if buffer_index is None or buffer_index == -1:
            return

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, int(buffer_index))
        gl.glUseProgram(program.program)

        # Position attribute
        if program.attribute_location_position != -1:
            gl.glEnableVertexAttribArray(program.attribute_location_position)
            gl.glVertexAttribPointer(
                program.attribute_location_position,
                program.attribute_size_position,
                gl.GL_FLOAT,
                False,
                program.attribute_stride_position,
                program.attribute_offset_position,
            )

        # Texture coordinates attribute
        if program.attribute_location_texture_coordinates != -1:
            gl.glEnableVertexAttribArray(program.attribute_location_texture_coordinates)
            gl.glVertexAttribPointer(
                program.attribute_location_texture_coordinates,
                program.attribute_size_texture_coordinates,
                gl.GL_FLOAT,
                False,
                program.attribute_stride_texture_coordinates,
                program.attribute_offset_texture_coordinates,
            )


    def unlink_buffer_from_shader_program(self, program: Optional[ShaderProgram]) -> None:
        if program is None or program.program == 0:
            return

        if program.attribute_location_texture_coordinates != -1:
            gl.glDisableVertexAttribArray(program.attribute_location_texture_coordinates)

        if program.attribute_location_position != -1:
            gl.glDisableVertexAttribArray(program.attribute_location_position)

    # ----------------------------------------------------------------------
    # Uniform helpers
    # ----------------------------------------------------------------------

    def uniforms_texture_size_set(self, program: Optional[ShaderProgram], width: float, height: float) -> None:
        if program is None:
            return
        if program.uniform_location_texture_size != -1:
            gl.glUniform2f(program.uniform_location_texture_size, float(width), float(height))

    # ModulateColor (from Color object)
    def uniforms_modulate_color_set_color(
        self,
        program: Optional[ShaderProgram],
        color: GraphicsColor,
    ) -> None:
        if program is None:
            return
        loc = program.uniform_location_modulate_color
        if loc != -1:
            gl.glUniform4f(loc, color.r, color.g, color.b, color.a)

    # ModulateColor (explicit RGBA)
    def uniforms_modulate_color_set(
        self,
        program: Optional[ShaderProgram],
        r: float,
        g: float,
        b: float,
        a: float,
    ) -> None:
        if program is None:
            return
        loc = program.uniform_location_modulate_color
        if loc != -1:
            gl.glUniform4f(loc, r, g, b, a)

    
    def uniforms_matrices_set_buffer(
        self,
        program: Optional[ShaderProgram],
        projection_buffer,
        model_view_buffer,
    ) -> None:
        if program is None:
            return

        projection_location = program.uniform_location_projection_matrix
        model_view_location = program.uniform_location_model_view_matrix

        # Projection
        if projection_location != -1:
            arr_p = np.asarray(projection_buffer, dtype=np.float32)
            if arr_p.size != 16:
                raise ValueError("Projection buffer must contain 16 floats")
            gl.glUniformMatrix4fv(projection_location, 1, False, arr_p)

        # ModelView
        if model_view_location != -1:
            arr_mv = np.asarray(model_view_buffer, dtype=np.float32)
            if arr_mv.size != 16:
                raise ValueError("Model-view buffer must contain 16 floats")
            gl.glUniformMatrix4fv(model_view_location, 1, False, arr_mv)
    
    def uniforms_matrices_set(
        self,
        program: Optional[ShaderProgram],
        projection_matrix: Optional[GraphicsMatrix],
        model_view_matrix: Optional[GraphicsMatrix],
    ) -> None:
        if program is None:
            return

        projection_location = program.uniform_location_projection_matrix
        model_view_location = program.uniform_location_model_view_matrix

        # Projection
        if projection_location != -1 and projection_matrix is not None:
            arr_p = np.asarray(projection_matrix.array(), dtype=np.float32)
            gl.glUniformMatrix4fv(projection_location, 1, False, arr_p)

        # ModelView
        if model_view_location != -1 and model_view_matrix is not None:
            arr_mv = np.asarray(model_view_matrix.array(), dtype=np.float32)
            gl.glUniformMatrix4fv(model_view_location, 1, False, arr_mv)
            
    def uniforms_texture_set_texture(
        self,
        program: Optional[ShaderProgram],
        texture: Optional[GraphicsTexture],
    ) -> None:
        if texture is None:
            return
        self.uniforms_texture_set_index(program=program, texture_index=texture.texture_index)

    def uniforms_texture_set_sprite(
        self,
        program: Optional[ShaderProgram],
        sprite: Optional[GraphicsSprite],
    ) -> None:
        if sprite is None:
            return
        self.uniforms_texture_set_texture(program, sprite.texture)

    def uniforms_texture_set_index(
        self,
        program: Optional[ShaderProgram],
        texture_index: int,
    ) -> None:
        if program is None:
            return
        loc = program.uniform_location_texture
        if loc == -1 or texture_index == -1:
            return
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_index)
        gl.glUniform1i(loc, 0)
