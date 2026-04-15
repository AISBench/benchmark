import base64
from io import BytesIO
from PIL import Image
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

def pil_to_base64(image, format="JPEG"):
    """
    Convert PIL Image to base64 string
    """
    if not isinstance(image, Image.Image):
        raise AISBenchDataContentError(UTILS_CODES.UNKNOWN_ERROR, "Input must be a PIL Image object")
    buffered = BytesIO()
    image.save(buffered, format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str