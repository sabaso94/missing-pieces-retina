"""The one preprocessing step every site goes through: crop to the fundus disc, pad to a square, resize.

Fundus photos from different cameras differ in size, aspect ratio and how much black border surrounds the
circular image of the retina. Cropping to the disc and resizing removes those differences, so that what is
left between sites is the retina itself (and the camera's colour and sharpness), not the framing. No colour
normalisation is applied: it would remove part of the real shift the study is about.
"""

import numpy as np
from PIL import Image

IMAGE_SIZE = 384


def crop_disc(img: Image.Image, threshold: int = 15) -> Image.Image:
    """Square crop around the fundus disc, padded with black where the disc is cut off by the frame.

    The disc is everything whose red channel is above `threshold` (the border is near black, the retina is
    red-orange). If almost nothing passes (a very dark image), the centre square is used instead.
    """
    rgb = img.convert("RGB")
    red = np.asarray(rgb)[:, :, 0]
    mask = red > threshold
    width, height = rgb.size
    if mask.mean() < 0.01:
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        return rgb.crop((left, top, left + side, top + side))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    side = max(bottom - top, right - left)
    # Centre the square on the disc's bounding box; PIL fills the parts outside the image with black.
    cx, cy = (left + right) // 2, (top + bottom) // 2
    return rgb.crop((cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side))


def prepare_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """RGB, cropped to the disc, resized to size x size."""
    return crop_disc(img.convert("RGB")).resize((size, size), Image.Resampling.LANCZOS)
