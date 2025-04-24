import base64
from io import BytesIO

from PIL import Image, ImageDraw

from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)


def resize_image_keep_aspect(img: Image.Image, long_size: int):
    # 縦横比を維持しつつ、長辺がlong_sizeになるようにリサイズします。
    width, height = img.size

    if height > width:
        new_height = long_size
        new_width = int(new_height * width / height)
    else:
        new_width = long_size
        new_height = int(new_width * height / width)

    # 画像を新しい解像度にリサイズします。
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img_resized


def base642pil(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    return image


def pil2base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def draw_bboxes(
    image: Image.Image, bboxes: list[tuple[int, int, int, int]], color: str = "green"
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=2)
    return image
