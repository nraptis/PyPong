# pong_net.py

from __future__ import annotations
from typing import List

from graphics.graphics_library import GraphicsLibrary
from graphics.graphics_pipeline import GraphicsPipeline
from graphics.graphics_matrix import GraphicsMatrix
from graphics.graphics_color import GraphicsColor
from graphics.graphics_shape_2d_instance import GraphicsShape2DInstance
from graphics.shader_program import ShaderProgram

class PongNet:
    def __init__(self) -> None:
        # All active net segments (vertical rectangles)
        self.net_instances: List[GraphicsShape2DInstance] = []

        # Net layout constants
        self.chunk_width: float = 18.0
        self.chunk_height: float = 34.0
        self.chunk_spacing: float = 16.0

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------
    def rebuild(self, graphics: GraphicsLibrary) -> None:
        self.dispose()

        width = float(graphics.frame_buffer_width)
        height = float(graphics.frame_buffer_height)

        max_net_height = min(width * 0.85, height - 64.0)

        # How many segments can we fit?
        segment_count = 1
        used_height = self.chunk_height

        # Try adding more segments while staying within max height
        while True:
            next_height = used_height + self.chunk_spacing + self.chunk_height
            if next_height > max_net_height:
                break

            used_height = next_height
            segment_count += 1

        # Actual total used height (segments + gaps between them)
        total_height = self.chunk_height * segment_count + self.chunk_spacing * (segment_count - 1)

        left = (width / 2.0) - (self.chunk_width / 2.0)
        top = (height / 2.0) - (total_height / 2.0)
        
        y = top
        for _ in range(segment_count):
            instance = GraphicsShape2DInstance()
            instance.load(graphics)
            instance.color = GraphicsColor(0.34, 0.34, 0.36, 1.0)
            instance.set_position_frame(
                x=left,
                y=y,
                width=self.chunk_width,
                height=self.chunk_height,
            )
            self.net_instances.append(instance)
            y += self.chunk_height + self.chunk_spacing

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def draw(
        self,
        graphics: GraphicsLibrary,
        pipeline: GraphicsPipeline,
        projection_matrix: GraphicsMatrix,
    ) -> None:
        for instance in self.net_instances:
            instance.projection_matrix = projection_matrix.copy()
            instance.render(pipeline.program_shape_2d)

    # ------------------------------------------------------------------
    # Dispose
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        """
        Dispose all GPU buffers for net segments
        and clear the segment list.
        Safe to call multiple times.
        """
        for seg in self.net_instances:
            seg.dispose()

        self.net_instances.clear()