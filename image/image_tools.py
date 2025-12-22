from __future__ import annotations
from PIL import Image
from typing import List, Tuple
import numpy as np
import torch

class ImageTools:

    # -----------------------------
    # Nearest-neighbor scaling
    # -----------------------------
    @classmethod
    def scale_nn(cls, img: Image.Image, factor: int) -> Image.Image:
        if factor <= 0:
            raise ValueError("scale factor must be >= 1")

        if factor == 1:
            return img

        w, h = img.size
        return img.resize(
            (w * factor, h * factor),
            resample=Image.NEAREST
        )
    
    @classmethod
    def scale(cls, img: Image.Image, scale: float) -> Image.Image:
        """
        Bilinear scaling by a float factor.
        Works with RGBA (and any PIL mode).
        """
        if scale <= 0:
            raise ValueError("scale must be > 0")

        if scale == 1.0:
            return img

        w, h = img.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        return img.resize(
            (new_w, new_h),
            resample=Image.BILINEAR
        )

    # -----------------------------
    # Public: normalize to RGBA
    # -----------------------------
    @classmethod
    def to_rgba(cls, img: Image.Image) -> Image.Image:
        """
        Convert any reasonable PIL image to RGBA.
        Alpha is preserved when present, otherwise set to 255.
        """
        if img.mode == "RGBA":
            return img

        if img.mode == "RGB":
            return img.convert("RGBA")

        if img.mode == "L":
            return img.convert("RGBA")

        if img.mode == "LA":
            return img.convert("RGBA")

        if img.mode == "P":
            return img.convert("RGBA")

        if img.mode == "CMYK":
            return img.convert("RGBA")

        # Fallback: let PIL decide
        return img.convert("RGBA")

    # -----------------------------
    # Grid composer (RGBA)
    # -----------------------------
    @classmethod
    def grid(
        cls,
        images: List[Image.Image],
        grid_width: int,
        grid_height: int,
        *,
        spacing_h: int = 8,
        spacing_v: int = 8,
        padding: int = 0,
        background_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> Image.Image:
        """
        Compose images into a fixed RGBA grid.

        All images are converted to RGBA.
        Images must all be the same size.
        """

        expected = grid_width * grid_height
        if len(images) != expected:
            raise ValueError(
                f"Expected {expected} images for grid "
                f"({grid_width} x {grid_height}), got {len(images)}"
            )

        # Normalize to RGBA
        imgs = [cls.to_rgba(im) for im in images]

        # Validate uniform size
        w, h = imgs[0].size
        for i, im in enumerate(imgs):
            if im.size != (w, h):
                raise ValueError(
                    f"Image {i} has size {im.size}, expected {(w, h)}"
                )

        total_w = (
            padding * 2
            + grid_width * w
            + (grid_width - 1) * spacing_h
        )
        total_h = (
            padding * 2
            + grid_height * h
            + (grid_height - 1) * spacing_v
        )

        canvas = Image.new(
            "RGBA",
            (total_w, total_h),
            background_color
        )

        idx = 0
        for gy in range(grid_height):
            for gx in range(grid_width):
                x = padding + gx * (w + spacing_h)
                y = padding + gy * (h + spacing_v)
                canvas.paste(imgs[idx], (x, y), imgs[idx])
                idx += 1

        return canvas
    
    @classmethod
    def pad(
        cls,
        img: Image.Image,
        top: int,
        right: int,
        bottom: int,
        left: int,
        background_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> Image.Image:
        # Validate padding
        for name, v in [("top", top), ("right", right), ("bottom", bottom), ("left", left)]:
            if v < 0:
                raise ValueError(f"pad {name} must be >= 0 (got {v})")

        # Normalize to RGBA
        img_rgba = cls.to_rgba(img)

        w, h = img_rgba.size
        new_w = w + left + right
        new_h = h + top + bottom

        canvas = Image.new(
            "RGBA",
            (new_w, new_h),
            background_color,
        )

        # Paste original image at offset
        canvas.paste(img_rgba, (left, top), img_rgba)

        return canvas

    @classmethod
    def pad_all(
        cls,
        img: Image.Image,
        padding: int,
        background_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> Image.Image:
        if padding < 0:
            raise ValueError(f"padding must be >= 0 (got {padding})")

        return cls.pad(
            img,
            top=padding,
            right=padding,
            bottom=padding,
            left=padding,
            background_color=background_color,
        )
    
    @classmethod
    def flip_h(cls, img: Image.Image) -> Image.Image:
        """Horizontal flip (mirror left-right)."""
        im = cls.to_rgba(img)
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    @classmethod
    def flip_v(cls, img: Image.Image) -> Image.Image:
        """Vertical flip (mirror top-bottom)."""
        im = cls.to_rgba(img)
        return im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    @classmethod
    def rotate_right(cls, img: Image.Image) -> Image.Image:
        """Rotate 90 degrees clockwise."""
        im = cls.to_rgba(img)
        return im.transpose(Image.Transpose.ROTATE_270)  # clockwise

    @classmethod
    def rotate_left(cls, img: Image.Image) -> Image.Image:
        """Rotate 90 degrees counterclockwise."""
        im = cls.to_rgba(img)
        return im.transpose(Image.Transpose.ROTATE_90)   # counterclockwise
    
    @classmethod
    def to_torch_chw_float(
        cls,
        image: Image.Image,
        *,
        channels: int = 3,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert PIL image to torch tensor [C,H,W] float32.

        channels:
          - 1  -> grayscale (L)
          - 3  -> RGB
          - 4  -> RGBA

        normalize:
          - True  -> values in [0,1]
          - False -> values in [0,255]
        """
        if channels not in (1, 3, 4):
            raise ValueError(f"channels must be 1, 3, or 4 (got {channels})")

        # ---- Mode conversion ----
        if channels == 1:
            if image.mode != "L":
                image = image.convert("L")
        elif channels == 3:
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:  # channels == 4
            if image.mode != "RGBA":
                image = image.convert("RGBA")

        # ---- PIL -> numpy ----
        arr = np.array(image, dtype=np.uint8, copy=True)

        # Ensure shape [H,W,C]
        if channels == 1:
            if arr.ndim != 2:
                raise ValueError(f"Expected grayscale image [H,W], got {arr.shape}")
            arr = arr[:, :, None]  # [H,W,1]
        else:
            if arr.ndim != 3 or arr.shape[2] != channels:
                raise ValueError(f"Expected image [H,W,{channels}], got {arr.shape}")

        # ---- numpy -> torch ----
        t = torch.from_numpy(arr).float()  # [H,W,C]

        if normalize:
            t = t.div(255.0)

        # ---- HWC -> CHW ----
        t = t.permute(2, 0, 1).contiguous()  # [C,H,W]

        return t

    @classmethod
    def to_torch_bchw_float(
        cls,
        image: Image.Image,
        *,
        channels: int = 3,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Convert PIL image to torch tensor [B,C,H,W] float32.
        Batch dimension B = 1.
        """
        t = cls.to_torch_chw_float(
            image,
            channels=channels,
            normalize=normalize,
        )  # [C,H,W]
        
        return t.unsqueeze(0)  # [1,C,H,W]