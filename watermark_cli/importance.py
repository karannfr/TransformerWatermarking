from typing import Optional
import os

import numpy as np

try:
	import torch
	exists_torch = True
except Exception:
	exists_torch = False

def _fallback_importance(img: np.ndarray) -> np.ndarray:
	# Simple gradient magnitude heuristic
	import cv2
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
	dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
	mag = np.sqrt(dx * dx + dy * dy)
	mag = mag / (mag.max() + 1e-6)
	return mag.astype(np.float32)


class ImportanceMap:
	def __init__(self, model_path: Optional[str] = None):
		self.model_path = model_path
		self.model = None
		if exists_torch and model_path and os.path.exists(model_path):
			try:
				from watermark_cli.models.transformer import ViTImportance
				self.model = ViTImportance.load_from_checkpoint(model_path)
				self.model.eval()
				torch.set_grad_enabled(False)
			except Exception:
				self.model = None

	def get_map(self, img: np.ndarray) -> np.ndarray:
		if self.model is None:
			return _fallback_importance(img)
		import torch
		arr = img.astype(np.float32) / 255.0
		t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
		with torch.no_grad():
			m = self.model(t)
		m = m.squeeze(0).squeeze(0).cpu().numpy()
		m = np.clip(m, 0.0, 1.0).astype(np.float32)
		return m
