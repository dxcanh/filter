import numpy as np
import base64
import cv2
from PIL import Image
import requests
import os

def loadBase64Img(uri):
    """Load image from base64 string.
    Args:
        uri: a base64 string.
    Returns:
        numpy array: the loaded image.
    """
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
    """Load image from path, url, base64 or numpy array.
    Args:
        img: a path, url, base64 or numpy array.
    Raises:
        ValueError: if the image path does not exist.
    Returns:
        numpy array: the loaded image.
    """
    # The image is already a numpy array
    if type(img).__module__ == np.__name__:
        return img

    # The image is a base64 string
    if img.startswith("data:image/"):
        return loadBase64Img(img)

    # The image is a url
    if img.startswith("http"):
        return np.array(Image.open(requests.get(img, stream=True, timeout=60).raw).convert("RGB"))[
            :, :, ::-1
        ]

    # The image is a path
    if os.path.isfile(img) is not True:
        raise ValueError(f"Confirm that {img} exists")

    return cv2.imread(img)

def load_pil_image(img):
    # the image is a Image object
    if isinstance(img, Image.Image):
        return img
    
    # the image is a numpy array
    if type(img).__module__ == np.__name__:
        return Image.fromarray(img)

    # The image is a path
    if os.path.isfile(img) is not True:
        raise ValueError(f"Confirm that {img} exists")
    return Image.open(img).convert('RGB')