# graphics_texture.py

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from graphics.graphics_library import GraphicsLibrary

from PIL import Image
import numpy as np
from OpenGL import GL as gl
from filesystem.file_utils import FileUtils
from typing import Optional

class GraphicsTexture:

    def __init__(
        self,
        graphics: Optional["GraphicsLibrary"] = None,
        subdirectory: Optional[str] = None,
        name: str = None):

        self.graphics: Optional["GraphicsLibrary"] = graphics
        self.subdirectory = subdirectory
        self.name = name
        self.texture_index: int = -1
        self.width: int = 0
        self.height: int = 0
        self.widthf: float = 0.0
        self.heightf: float = 0.0

        # Auto-load if both graphics + path are provided
        if graphics is not None:
            self.load()

    # --------------------------------------------------------------
    # Load/reload texture from file
    # --------------------------------------------------------------
    def load(self) -> None:
        """
        Load or reload the texture from file_name.
        """

        # If previously loaded, delete old GL texture
        self.dispose()

        if self.graphics is None or self.name is None:
            return

        try:
            image = FileUtils.load_local_pillow_image(self.subdirectory, self.name)
            image_rgba = image.convert("RGBA")
        except Exception:
            # If something weird comes back, treat as "no texture"
            self.texture_index = -1
            return
        
        self.width, self.height = image_rgba.size
        self.widthf = float(self.width)
        self.heightf = float(self.height)

        bitmap = np.array(image_rgba, dtype=np.uint8)

        texture_index = self.graphics.texture_generate_from_bitmap(bitmap)
        # Normalize to plain Python int in case it's a numpy scalar/array
        try:
            self.texture_index = int(texture_index)
        except Exception:
            # If something weird comes back, treat as "no texture"
            self.texture_index = -1

    # --------------------------------------------------------------
    # Unload / delete GPU texture
    # --------------------------------------------------------------
    def dispose(self) -> None:
        """
        Delete the texture from GPU and reset fields.
        Safe to call multiple times.
        """
        if self.graphics is not None and self.texture_index is not None:
            # texture_delete already guards None / -1 and weird types
            self.graphics.texture_delete(self.texture_index)

        self.texture_index = -1
        self.width = 0
        self.height = 0
        self.widthf = 0.0
        self.heightf = 0.0

    def print(self) -> None:
        print("GraphicsTexture -> [" + str(self.width) + ", " + str(self.height) + "]")
        print("\tIndex = " + str(self.texture_index))
        if self.subdirectory:
             print("\tFile: " + str(self.subdirectory) + " / " + self.name)
        else:
            print("\tFile: " + self.name)
             

       
