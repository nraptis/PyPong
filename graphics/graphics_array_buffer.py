# graphics_array_buffer.py

from __future__ import annotations
from typing import Generic, TypeVar, Optional, Sequence, TYPE_CHECKING
from graphics.graphics_float_bufferable import GraphicsFloatBufferable
if TYPE_CHECKING:
    from graphics.graphics_library import GraphicsLibrary

T = TypeVar("T", bound=GraphicsFloatBufferable)

class GraphicsArrayBuffer(Generic[T]):
    """
    Statically allocated graphics buffer.
    The content can be replaced, but it cannot change size.
    """
    
    def __init__(self) -> None:
        self.graphics: Optional["GraphicsLibrary"] = None
        self.vertex_buffer: list[float] = []
        self.buffer_index: int = -1
        self.size: int = 0  # in bytes

    def load(self, graphics: "GraphicsLibrary", items: Sequence[T]) -> None:
        """
        Initialize the buffer from a sequence of GraphicsFloatBufferable items.
        """
        self.graphics = graphics
        if not items:
            self.vertex_buffer = []
            self.buffer_index = -1
            self.size = 0
            return

        # Total float count, then bytes
        float_count = graphics.buffer_float_size(items)
        self.size = float_count * 4  # 4 bytes per float

        # Build vertex data as Python list[float]
        self.vertex_buffer = graphics.buffer_float_generate_from_array(items)

        # Create VBO and upload
        self.buffer_index = graphics.buffer_array_generate()
        graphics.buffer_array_write(self.buffer_index, self.vertex_buffer)

    def write(self, items: Sequence[T]) -> None:
        """
        Overwrite the existing buffer contents with new items.
        Size must not change.
        """
        if self.graphics is None or self.buffer_index == -1:
            return
        graphics = self.graphics

        # Rebuild vertex data
        new_data: list[float] = []
        graphics.buffer_float_write_from_list(items, new_data)

        # Optional sanity check: same number of floats
        old_floats = self.size // 4 if self.size > 0 else 0
        new_floats = len(new_data)
        if old_floats != 0 and new_floats != old_floats:
            print(
                f"[GraphicsArrayBuffer] WARNING: write() changed float count "
                f"from {old_floats} to {new_floats}. Uploading anyway."
            )

        self.vertex_buffer = new_data
        graphics.buffer_array_write(self.buffer_index, self.vertex_buffer)

    def dispose(self) -> None:
        """
        Delete the GPU buffer and reset local state.
        Safe to call multiple times.
        """
        if self.graphics is not None:
            # Delete buffer if valid
            self.graphics.buffer_array_delete(self.buffer_index)

        # Reset state
        self.buffer_index = -1
        self.vertex_buffer.clear()
        self.size = 0
        self.graphics = None