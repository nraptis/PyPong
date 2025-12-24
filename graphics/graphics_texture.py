from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from graphics.graphics_library import GraphicsLibrary

from image.bitmap import Bitmap
import numpy as np
from PIL import Image
from filesystem.file_utils import FileUtils


class GraphicsTexture:

    def __init__(self, graphics: Optional["GraphicsLibrary"] = None) -> None:
        self.graphics: Optional["GraphicsLibrary"] = graphics

        self.texture_index: int = -1
        self.width: int = 0
        self.height: int = 0
        self.widthf: float = 0.0
        self.heightf: float = 0.0
        self.name = None
        self.subdirectory = None

    # --------------------------------------------------------------
    # Load from file (generate + write)
    # RULE: First line is always self.dispose()
    # Any failure => return (object remains disposed)
    # --------------------------------------------------------------
    def load_file(self, subdirectory: Optional[str], name: Optional[str]) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_file: graphics is None")
            return

        if name is None:
            print("⚠️ GraphicsTexture.load_file: name is None")
            return

        try:
            image = FileUtils.load_local_pillow_image(subdirectory, name)
        except Exception as e:
            print(
                "⚠️ GraphicsTexture.load_file: failed to load image "
                f"(subdirectory={subdirectory}, name={name}) | {e}"
            )
            return

        self.load_pillow(image)

    # --------------------------------------------------------------
    # Loaders (generate + write)
    # RULE: First line is always self.dispose()
    # Any failure => return (object remains disposed)
    # --------------------------------------------------------------
    def load_bitmap(self, bitmap: Optional[Bitmap]) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_bitmap: graphics is None")
            return
        if bitmap is None:
            print("⚠️ GraphicsTexture.load_bitmap: bitmap is None")
            return
        if bitmap.width <= 0 or bitmap.height <= 0:
            print(
                "⚠️ GraphicsTexture.load_bitmap: invalid bitmap size "
                f"width={bitmap.width}, height={bitmap.height}"
            )
            return

        texture_index = self.graphics.texture_generate_bitmap(bitmap)
        if (texture_index is None) or (texture_index == -1):
            print("⚠️ GraphicsTexture.load_bitmap: texture_generate_bitmap failed")
            return

        self.texture_index = int(texture_index)
        self._set_size(bitmap.width, bitmap.height)

    def load_pillow(self, image: Optional[Image.Image]) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_pillow: graphics is None")
            return
        if image is None:
            print("⚠️ GraphicsTexture.load_pillow: image is None")
            return

        width, height = image.size
        if width <= 0 or height <= 0:
            print(
                "⚠️ GraphicsTexture.load_pillow: invalid image size "
                f"width={width}, height={height}"
            )
            return

        texture_index = self.graphics.texture_generate_pillow(image)
        if (texture_index is None) or (texture_index == -1):
            print("⚠️ GraphicsTexture.load_pillow: texture_generate_pillow failed")
            return

        self.texture_index = int(texture_index)
        self._set_size(width, height)

    def load_numpy(self, data: Optional[np.ndarray], width: int, height: int) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_numpy: graphics is None")
            return
        if data is None:
            print("⚠️ GraphicsTexture.load_numpy: data is None")
            return

        try:
            width = int(width)
            height = int(height)
        except Exception:
            print(f"⚠️ GraphicsTexture.load_numpy: width/height not castable (width={width}, height={height})")
            return

        if width <= 0 or height <= 0:
            print(f"⚠️ GraphicsTexture.load_numpy: invalid size width={width}, height={height}")
            return

        texture_index = self.graphics.texture_generate_numpy(data, width, height)
        if (texture_index is None) or (texture_index == -1):
            print("⚠️ GraphicsTexture.load_numpy: texture_generate_numpy failed")
            return

        self.texture_index = int(texture_index)
        self._set_size(width, height)

    def load_random(self, width: int, height: int) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_random: graphics is None")
            return

        try:
            width = int(width)
            height = int(height)
        except Exception:
            print(f"⚠️ GraphicsTexture.load_random: width/height not castable (width={width}, height={height})")
            return

        if width <= 0 or height <= 0:
            print(f"⚠️ GraphicsTexture.load_random: invalid size width={width}, height={height}")
            return

        texture_index = self.graphics.texture_generate_random(width, height)
        if (texture_index is None) or (texture_index == -1):
            print("⚠️ GraphicsTexture.load_random: texture_generate_random failed")
            return

        self.texture_index = int(texture_index)
        self._set_size(width, height)

    def load_black(self, width: int, height: int) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_black: graphics is None")
            return

        try:
            width = int(width)
            height = int(height)
        except Exception:
            print(f"⚠️ GraphicsTexture.load_black: width/height not castable (width={width}, height={height})")
            return

        if width <= 0 or height <= 0:
            print(f"⚠️ GraphicsTexture.load_black: invalid size width={width}, height={height}")
            return

        count = 4 * width * height
        data = np.zeros((count,), dtype=np.uint8)
        data[3::4] = 255

        # Calls another load_* which disposes first; that's fine (idempotent).
        self.load_numpy(data, width, height)

    def load_white(self, width: int, height: int) -> None:
        self.dispose()

        if self.graphics is None:
            print("⚠️ GraphicsTexture.load_white: graphics is None")
            return

        try:
            width = int(width)
            height = int(height)
        except Exception:
            print(f"⚠️ GraphicsTexture.load_white: width/height not castable (width={width}, height={height})")
            return

        if width <= 0 or height <= 0:
            print(f"⚠️ GraphicsTexture.load_white: invalid size width={width}, height={height}")
            return

        count = 4 * width * height
        data = np.full((count,), 255, dtype=np.uint8)

        # Calls another load_* which disposes first; that's fine (idempotent).
        self.load_numpy(data, width, height)

    # --------------------------------------------------------------
    # Updating pixels later (write into existing texture)
    # These do NOT dispose. They require an existing texture.
    # --------------------------------------------------------------
    def write_numpy(self, data: Optional[np.ndarray]) -> None:
        if self.graphics is None:
            print("⚠️ GraphicsTexture.write_numpy: graphics is None")
            return
        if self.texture_index < 0:
            print("⚠️ GraphicsTexture.write_numpy: texture_index is invalid (-1)")
            return
        if self.width <= 0 or self.height <= 0:
            print("⚠️ GraphicsTexture.write_numpy: texture has invalid size")
            return

        self.graphics.texture_write_numpy(self.texture_index, data, self.width, self.height)

    def write_pillow(self, image: Optional[Image.Image]) -> None:
        if self.graphics is None:
            print("⚠️ GraphicsTexture.write_pillow: graphics is None")
            return
        if self.texture_index < 0:
            print("⚠️ GraphicsTexture.write_pillow: texture_index is invalid (-1)")
            return

        self.graphics.texture_write_pillow(self.texture_index, image)

    def write_bitmap(self, bitmap: Optional[Bitmap]) -> None:
        if self.graphics is None:
            print("⚠️ GraphicsTexture.write_bitmap: graphics is None")
            return
        if self.texture_index < 0:
            print("⚠️ GraphicsTexture.write_bitmap: texture_index is invalid (-1)")
            return

        self.graphics.texture_write_bitmap(self.texture_index, bitmap)

    # --------------------------------------------------------------
    # Unload / delete GPU texture
    # --------------------------------------------------------------
    def dispose(self) -> None:
        if self.graphics is not None:
            self.graphics.texture_delete(self.texture_index)

        self.texture_index = -1
        self.width = 0
        self.height = 0
        self.widthf = 0.0
        self.heightf = 0.0
        self.name = None
        self.subdirectory = None

    def _set_size(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.widthf = float(self.width)
        self.heightf = float(self.height)

    def print(self) -> None:
        print(f"GraphicsTexture -> [{self.width}, {self.height}]")
        print(f"\tIndex = {self.texture_index}")
        if self.subdirectory is not None or self.name is not None:
            if self.subdirectory is not None and self.name is not None:
                print(f"\tSource = {self.subdirectory}/{self.name}")
            elif self.subdirectory is not None:
                print(f"\tSource = {self.subdirectory}")
            else:
                print(f"\tSource = {self.name}")
