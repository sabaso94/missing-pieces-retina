from PIL import Image, ImageDraw

from retina_shift.preprocess import IMAGE_SIZE, crop_disc, prepare_image


def _disc_image(width=600, height=400, centre=(200, 200), radius=100):
    img = Image.new("RGB", (width, height), "black")
    x, y = centre
    ImageDraw.Draw(img).ellipse([x - radius, y - radius, x + radius, y + radius], fill=(200, 80, 40))
    return img


def test_crop_disc_finds_offcentre_disc():
    out = crop_disc(_disc_image())
    assert out.size[0] == out.size[1]
    assert abs(out.size[0] - 200) <= 4


def test_crop_disc_pads_a_cut_off_disc_to_square():
    # Disc wider than the image is tall (top and bottom cut off, as in many EyePACS photos).
    out = crop_disc(_disc_image(width=800, height=300, centre=(400, 150), radius=200))
    assert out.size[0] == out.size[1]
    assert abs(out.size[0] - 400) <= 4


def test_crop_disc_dark_image_falls_back_to_centre_square():
    assert crop_disc(Image.new("RGB", (600, 400), "black")).size == (400, 400)


def test_prepare_image_is_rgb_384():
    out = prepare_image(_disc_image().convert("L"))
    assert IMAGE_SIZE == 384
    assert out.size == (384, 384) and out.mode == "RGB"
