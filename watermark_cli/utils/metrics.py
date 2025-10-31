import numpy as np
from skimage.metrics import structural_similarity as ssim_metric


def compute_psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
	ref = ref.astype(np.float32)
	test = test.astype(np.float32)
	mse = np.mean((ref - test) ** 2)
	if mse <= 1e-12:
		return 99.0
	psnr = 10.0 * np.log10((data_range ** 2) / mse)
	return float(psnr)


def compute_ssim(ref: np.ndarray, test: np.ndarray) -> float:
	if ref.ndim == 3 and ref.shape[2] == 3:
		val = ssim_metric(ref, test, channel_axis=2, data_range=255)
	else:
		val = ssim_metric(ref, test, data_range=255)
	return float(val)
