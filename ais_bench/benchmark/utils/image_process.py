import base64
from io import BytesIO
from PIL import Image

def pil_to_base64(image, format="JPEG"):
    """
    Convert PIL Image to base64 string
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    buffered = BytesIO()
    image.save(buffered, format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str