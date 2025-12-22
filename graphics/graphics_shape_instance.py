# graphics_shape_instance.py

from __future__ import annotations
from typing import Generic, TypeVar, Sequence, Optional, Protocol

from graphics.graphics_primitives import PositionConforming2D
from graphics.graphics_float_bufferable import GraphicsFloatBufferable
from graphics.graphics_array_buffer import GraphicsArrayBuffer
from graphics.graphics_library import GraphicsLibrary
from graphics.graphics_matrix import GraphicsMatrix
from graphics.shader_program import ShaderProgram
from graphics.graphics_color import GraphicsColor

class ShapeVertexConforming(
    PositionConforming2D,
    GraphicsFloatBufferable,
    Protocol,
):
    ...

T = TypeVar("T", bound=ShapeVertexConforming)

class GraphicsShapeInstance(Generic[T]):
    def __init__(self, vertex_array: Sequence[T]) -> None:
        if len(vertex_array) != 4:
            raise ValueError("GraphicsShapeInstance requires exactly 4 vertices")

        self.vertex_array: list[T] = list(vertex_array)
        self.graphics_array_buffer: GraphicsArrayBuffer[T] = GraphicsArrayBuffer()

        self.graphics: Optional[GraphicsLibrary] = None

        self.indices = [0, 1, 2, 3]
        self.index_buffer: Optional[object] = None

        self.projection_matrix: GraphicsMatrix = GraphicsMatrix()
        self.model_view_matrix: GraphicsMatrix = GraphicsMatrix()

        self.color: GraphicsColor = GraphicsColor(1.0, 1.0, 1.0, 1.0)
        self.is_vertex_buffer_dirty: bool = True

    # ------------------------------------------------------------------
    # Vertex data manipulation
    # ------------------------------------------------------------------
    def set_vertex_array(self, vertex_array: Sequence[T]) -> None:
        """
        Replace the four vertices, preserving the same list object so the
        GraphicsArrayBuffer can keep referencing it.
        """
        if len(vertex_array) != 4:
            raise ValueError("set_vertex_array requires exactly 4 vertices")

        self.vertex_array[0] = vertex_array[0]
        self.vertex_array[1] = vertex_array[1]
        self.vertex_array[2] = vertex_array[2]
        self.vertex_array[3] = vertex_array[3]
        self.is_vertex_buffer_dirty = True

    def set_position_frame(self, x: float, y: float, width: float, height: float) -> None:
        """
        Convenience: set vertex positions to a rectangle at (x, y) with size (width, height).
        """
        self.set_position_quad(
            x,         y,
            x + width, y,
            x,         y + height,
            x + width, y + height,
        )

    def set_position_quad(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
        x4: float, y4: float,
    ) -> None:
        """
        Set explicit positions for all 4 vertices.
        """
        self.vertex_array[0].x = x1
        self.vertex_array[0].y = y1
        self.vertex_array[1].x = x2
        self.vertex_array[1].y = y2
        self.vertex_array[2].x = x3
        self.vertex_array[2].y = y3
        self.vertex_array[3].x = x4
        self.vertex_array[3].y = y4
        self.is_vertex_buffer_dirty = True

    # ------------------------------------------------------------------
    # Load / GPU setup
    # ------------------------------------------------------------------
    def load(self, graphics: Optional[GraphicsLibrary]) -> None:
        """
        Bind this instance to a GraphicsLibrary.
        Creates the index buffer and uploads the initial vertex data.
        """
        self.graphics = graphics

        if self.graphics is None:
            raise ValueError("GraphicsShapeInstance.load: graphics is None")

        # Upload buffers
        self.index_buffer = self.graphics.buffer_index_generate_from_list(self.indices)
        
        self.graphics_array_buffer.load(self.graphics, self.vertex_array)
        self.is_vertex_buffer_dirty = False

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, shader_program: Optional[ShaderProgram]) -> None:
        if shader_program is None:
            return
        if self.graphics is None:
            return
        if self.index_buffer is None:
            return
        
        graphics = self.graphics
        program = shader_program

        # Upload vertex array if modified
        if self.is_vertex_buffer_dirty:
            self.graphics_array_buffer.write(self.vertex_array)
            self.is_vertex_buffer_dirty = False

        # Attach vertex buffer to program
        graphics.link_buffer_to_shader_program(program, self.graphics_array_buffer)

        # Set uniforms (no texture)
        graphics.uniforms_modulate_color_set_color(program, self.color)
        graphics.uniforms_matrices_set(
            program=program,
            projection_matrix=self.projection_matrix,
            model_view_matrix=self.model_view_matrix,
        )

        # Draw as triangle strip with 4 indices
        graphics.draw_triangle_strips(self.index_buffer, 4)

        # Unbind
        graphics.unlink_buffer_from_shader_program(program)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        """
        Delete GPU buffers and reset bindings.
        Safe to call multiple times.
        """
        # Dispose vertex buffer (VBO)
        if self.graphics_array_buffer is not None:
            self.graphics_array_buffer.dispose()

        # Dispose index buffer (EBO) if valid
        if self.graphics is not None:
            self.graphics.buffer_index_delete(self.index_buffer)

        # Reset state
        self.index_buffer = None
        self.graphics = None
        self.is_vertex_buffer_dirty = True
