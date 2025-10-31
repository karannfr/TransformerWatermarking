import numpy as np
from PIL import Image


def load_image_as_numpy(path: str) -> np.ndarray:
	img = Image.open(path).convert("RGB")
	arr = np.array(img, dtype=np.uint8)
	return arr


def save_numpy_as_image(arr: np.ndarray, path: str) -> None:
	img = Image.fromarray(arr.astype(np.uint8))
	img.save(path)
