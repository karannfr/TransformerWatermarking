from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
	def __init__(self, in_chans: int = 3, embed_dim: int = 192, patch_size: int = 8):
		super().__init__()
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.patch_size = patch_size

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# B C H W -> B N D
		x = self.proj(x)
		b, d, h, w = x.shape
		x = x.flatten(2).transpose(1, 2)
		return x, (h, w)


class TransformerEncoder(nn.Module):
	def __init__(self, dim: int, depth: int = 4, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
		super().__init__()
		layers = []
		for _ in range(depth):
			layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio), dropout=dropout, batch_first=True, activation="gelu"))
		self.encoder = nn.TransformerEncoder(layers[0], num_layers=depth)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.encoder(x)


class ViTImportance(nn.Module):
	def __init__(self, patch_size: int = 8, embed_dim: int = 192, depth: int = 4, heads: int = 4):
		super().__init__()
		self.patch = PatchEmbed(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)
		self.pos = None
		self.tr = TransformerEncoder(dim=embed_dim, depth=depth, heads=heads)
		self.head = nn.Linear(embed_dim, 1)
		self.patch_size = patch_size

	@staticmethod
	def load_from_checkpoint(path: str) -> "ViTImportance":
		state = torch.load(path, map_location="cpu")
		cfg = state.get("__cfg__", {"patch_size": 8, "embed_dim": 192, "depth": 4, "heads": 4})
		model = ViTImportance(**cfg)
		model.load_state_dict(state["state_dict"])
		return model

	def save_checkpoint(self, path: str) -> None:
		state = {"state_dict": self.state_dict(), "__cfg__": {"patch_size": self.patch_size, "embed_dim": self.head.in_features, "depth": 4, "heads": 4}}
		torch.save(state, path)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: B C H W in [0,1]
		xp, (h, w) = self.patch(x)
		xp = self.tr(xp)
		logits = self.head(xp)  # B N 1
		logits = logits.transpose(1, 2).reshape(x.shape[0], 1, h, w)
		# Upsample to input size
		out = F.interpolate(logits, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
		out = torch.sigmoid(out)
		return out


def sobel_pseudolabels(x: torch.Tensor) -> torch.Tensor:
	# x: B C H W in [0,1]
	import cv2
	b, c, h, w = x.shape
	outs = []
	for i in range(b):
		img = (x[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
		dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
		mag = (np.sqrt(dx * dx + dy * dy))
		mag = mag / (mag.max() + 1e-6)
		outs.append(mag)
	arr = np.stack(outs, axis=0)
	arr = arr[:, None, :, :]
	return torch.from_numpy(arr).to(x.device).float()
