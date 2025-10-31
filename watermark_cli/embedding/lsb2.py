import hashlib
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from watermark_cli.security.keystream import keyed_permutation


def select_positions(img_shape: Tuple[int, int, int], num_bits: int, key: str, channels: str = "rgb", importance: Optional[np.ndarray] = None) -> List[Tuple[int, int, int]]:
	"""Select (row, col, channel) positions to write bits.
	Each position stores 2 bits (LSB2). We therefore select ceil(num_bits/2) positions.
	Generation avoids building the full image coordinate list to reduce memory usage.
	"""
	height, width, channels_n = img_shape
	# Determine channel indices
	if channels.lower() == "rgb":
		ch_indices = [0, 1, 2][:channels_n]
	elif channels.lower() == "r":
		ch_indices = [0]
	elif channels.lower() == "g":
		ch_indices = [1]
	elif channels.lower() == "b":
		ch_indices = [2]
	else:
		ch_indices = [0, 1, 2][:channels_n]

	positions_needed = (num_bits + 1) // 2  # 2 bits per position

	# Normalize or create importance map at image size
	if importance is None:
		importance = np.ones((height, width), dtype=np.float32)
	else:
		imp = importance.astype(np.float32)
		if imp.shape[:2] != (height, width):
			imp = np.array(Image.fromarray((np.clip(imp, 0.0, 1.0) * 255).astype(np.uint8)).resize((width, height)))
			imp = imp.astype(np.float32) / 255.0
		importance = np.clip(imp, 0.0, 1.0)

	# Quantize importance to 8-bit before ranking to avoid tiny-order flips
	impq = (np.clip(importance, 0.0, 1.0) * 255.0).astype(np.uint8)
	flat_idx = np.argsort(-impq.reshape(-1))

	# We only need enough pixels to supply positions_needed across chosen channels
	per_pixel_capacity = len(ch_indices)
	pixels_needed = (positions_needed + per_pixel_capacity - 1) // per_pixel_capacity
	# Add a small margin to allow key-based shuffling without running out
	pixels_needed = min(pixels_needed + 1024, height * width)

	top_pixels = flat_idx[:pixels_needed]
	# Build coords only from the top subset, then shuffle and take exactly positions_needed
	coords = []
	for i in top_pixels:
		r, c = divmod(int(i), width)
		for ch in ch_indices:
			coords.append((r, c, ch))

	coords = keyed_permutation(coords, key)
	return coords[:positions_needed]


def embed_bits_lsb2(img: np.ndarray, bits: np.ndarray, positions: List[Tuple[int, int, int]]) -> np.ndarray:
	"""Embed bits using 2 LSBs per channel position. Expects arbitrary bit length.
	Uses positions_count * 2 bits at most; excess bits are ignored. If odd length, last high bit is zero-padded.
	"""
	out = img.copy()
	bits = bits.astype(np.uint8)
	for i, (r, c, ch) in enumerate(positions):
		b0_idx = 2 * i
		b1_idx = 2 * i + 1
		if b0_idx >= bits.size:
			break
		b0 = int(bits[b0_idx])
		b1 = int(bits[b1_idx]) if b1_idx < bits.size else 0
		val = int(out[r, c, ch])
		val = (val & ~0b11) | ((b1 << 1) | b0)
		out[r, c, ch] = val
	return out


def extract_bits_lsb2(img: np.ndarray, positions: List[Tuple[int, int, int]]) -> np.ndarray:
	"""Extract 2 LSBs per position and return a flat bit array of length 2*len(positions)."""
	out_bits = np.zeros(len(positions) * 2, dtype=np.uint8)
	for i, (r, c, ch) in enumerate(positions):
		val = int(img[r, c, ch])
		b0 = val & 0b1
		b1 = (val >> 1) & 0b1
		out_bits[2 * i] = b0
		out_bits[2 * i + 1] = b1
	return out_bits
