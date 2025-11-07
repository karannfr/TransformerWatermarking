import os
import json
import math
import base64
from typing import Optional

import click
import numpy as np
from PIL import Image
import cv2

from watermark_cli.embedding.lsb2 import embed_bits_lsb2, extract_bits_lsb2, select_positions
from watermark_cli.utils.image_io import load_image_as_numpy, save_numpy_as_image
from watermark_cli.utils.metrics import compute_psnr, compute_ssim
from watermark_cli.importance import ImportanceMap


def _text_to_bits(text: str) -> np.ndarray:
	data = text.encode("utf-8")
	bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
	return bits.astype(np.uint8)


def _bits_to_text(bits: np.ndarray) -> str:
	n = (bits.size // 8) * 8
	bits = bits[:n]
	if bits.size == 0:
		return ""
	arr = np.packbits(bits.astype(np.uint8))
	try:
		return arr.tobytes().decode("utf-8", errors="ignore")
	except Exception:
		return ""


def _b64_to_bits(b64: str) -> np.ndarray:
	raw = base64.b64decode(b64)
	return np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(np.uint8)


def _bits_to_b64(bits: np.ndarray) -> str:
	n = (bits.size // 8) * 8
	bits = bits[:n]
	arr = np.packbits(bits.astype(np.uint8))
	return base64.b64encode(arr.tobytes()).decode("ascii")


@click.group()
def cli():
	"""Watermark CLI: Transformer + ECC + LSB2"""
	pass


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--message", "message", required=False, type=str)
@click.option("--message-b64", "message_b64", required=False, type=str)
@click.option("--key", "key", required=True, type=str)
@click.option("--channels", default="rgb", show_default=True, type=str)
@click.option("--ecc-symbols", default=64, show_default=True, type=int)
@click.option("--model", "model_path", default="./checkpoints/vit_sobel.pth", show_default=True, type=str)
@click.option("--use-importance", "use_importance", is_flag=True, default=False, help="Use transformer importance map")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output JSON")

def embed(input_path: str, output_path: str, message: Optional[str], message_b64: Optional[str], key: str, channels: str, ecc_symbols: int, model_path: str, use_importance: bool, as_json: bool):
	"""Embed a text or binary watermark into an image."""
	if not message and not message_b64:
		raise click.UsageError("Provide --message or --message-b64")

	payload_bits = _b64_to_bits(message_b64) if message_b64 else _text_to_bits(message or "")
	# Import ECC codec lazily so commands that don't need it (like diff/quality) don't require reedsolo
	try:
		from watermark_cli.ecc.rs import ecc_encode_bits
	except Exception as e:
		raise click.ClickException("Missing dependency for ECC encoding. Install requirements.txt (pip install -r requirements.txt) or install 'reedsolo'. Error: %s" % e)
	coded_bits = ecc_encode_bits(payload_bits, rs_parity_symbols=ecc_symbols)
	img = load_image_as_numpy(input_path)

	importance = None
	if use_importance:
		# Stabilize importance input by zeroing 2 LSBs so embed/extract see the same content
		stable = (img & 0b11111100).astype(np.uint8)
		imp = ImportanceMap(model_path=model_path)
		importance = imp.get_map(stable)

	positions = select_positions(img.shape, len(coded_bits), key=key, channels=channels, importance=importance)
	stego = embed_bits_lsb2(img, coded_bits, positions)
	save_numpy_as_image(stego, output_path)
	if as_json:
		click.echo(json.dumps({"status": "ok", "output": output_path, "bits": int(coded_bits.size)}))
	else:
		click.echo("ok")


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--key", "key", required=True, type=str)
@click.option("--length", "length_bits", required=True, type=int)
@click.option("--channels", default="rgb", show_default=True, type=str)
@click.option("--ecc-symbols", default=64, show_default=True, type=int)
@click.option("--model", "model_path", default="./checkpoints/vit_sobel.pth", show_default=True, type=str)
@click.option("--use-importance", "use_importance", is_flag=True, default=False, help="Use transformer importance map")
@click.option("--as-b64/--as-text", "as_b64", default=False, show_default=True)
@click.option("--print", "print_out", is_flag=True, default=True)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output JSON")

def extract(input_path: str, key: str, length_bits: int, channels: str, ecc_symbols: int, model_path: str, use_importance: bool, as_b64: bool, print_out: bool, as_json: bool):
	"""Extract a watermark from an image."""
	img = load_image_as_numpy(input_path)
	payload_bytes = (length_bits + 7) // 8
	coded_bytes = payload_bytes + ecc_symbols
	coded_bits_len = coded_bytes * 8

	importance = None
	if use_importance:
		# Stabilize importance input by zeroing 2 LSBs so embed/extract see the same content
		stable = (img & 0b11111100).astype(np.uint8)
		imp = ImportanceMap(model_path=model_path)
		importance = imp.get_map(stable)

	positions = select_positions(img.shape, coded_bits_len, key=key, channels=channels, importance=importance)
	recovered_bits = extract_bits_lsb2(img, positions)[:coded_bits_len]
	# Import ECC decode lazily
	try:
		from watermark_cli.ecc.rs import ecc_decode_bits
	except Exception as e:
		raise click.ClickException("Missing dependency for ECC decoding. Install requirements.txt (pip install -r requirements.txt) or install 'reedsolo'. Error: %s" % e)
	decoded_bits = ecc_decode_bits(recovered_bits, expected_payload_bits=length_bits, rs_parity_symbols=ecc_symbols)

	if decoded_bits is None:
		if as_json:
			click.echo(json.dumps({"status": "fail", "error": "ECC decode failed"}))
		else:
			click.echo("decode failed")
		return

	msg = _bits_to_b64(decoded_bits) if as_b64 else _bits_to_text(decoded_bits)
	if as_json:
		click.echo(json.dumps({"status": "ok", "message": msg}))
	else:
		click.echo(msg)


@cli.command()
@click.option("--ref", "ref_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--test", "test_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, default=False, help="Output JSON")

def quality(ref_path: str, test_path: str, as_json: bool):
	"""Compute quality metrics (PSNR, SSIM)."""
	ref = load_image_as_numpy(ref_path)
	test = load_image_as_numpy(test_path)
	psnr = compute_psnr(ref, test)
	ssim = compute_ssim(ref, test)
	if as_json:
		click.echo(json.dumps({"psnr": float(psnr), "ssim": float(ssim)}))
	else:
		click.echo(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")


def _save_diff_images(ref: np.ndarray, test: np.ndarray, out_dir: str, base_name: str) -> dict:
	"""Create and save visualizations comparing ref and test images.

	Saves three images in out_dir with base_name prefix:
	- <base_name>_absdiff.png : per-channel absolute difference image
	- <base_name>_heatmap.png : colored heatmap of difference overlaid on test image
	- <base_name>_sidebyside.png : reference | test | heatmap overlay

	Returns dict with saved paths.
	"""
	out_dir = out_dir or "."
	os.makedirs(out_dir, exist_ok=True)
	paths = {}
	# Absolute per-channel difference (clipped to 0-255)
	absdiff = np.abs(ref.astype(np.int16) - test.astype(np.int16)).astype(np.uint8)
	abs_path = os.path.join(out_dir, f"{base_name}_absdiff.png")
	Image.fromarray(absdiff).save(abs_path)
	paths["absdiff"] = abs_path

	# Per-pixel magnitude (L2) for heatmap
	mag = np.linalg.norm((ref.astype(np.float32) - test.astype(np.float32)), axis=2)
	if mag.max() <= 0.0:
		norm = np.zeros_like(mag, dtype=np.uint8)
	else:
		norm = (255.0 * (mag / float(mag.max()))).clip(0, 255).astype(np.uint8)

	# Apply OpenCV colormap (expects single-channel 8-bit), returns BGR
	heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
	# Convert heatmap to RGB to match our arrays
	heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

	# Overlay heatmap onto test image (alpha blend)
	alpha = 0.5
	overlay = (test.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
	heat_path = os.path.join(out_dir, f"{base_name}_heatmap.png")
	Image.fromarray(overlay).save(heat_path)
	paths["heatmap_overlay"] = heat_path

	# Side-by-side: ref | test | overlay
	sb = np.concatenate([ref, test, overlay], axis=1)
	sb_path = os.path.join(out_dir, f"{base_name}_sidebyside.png")
	Image.fromarray(sb).save(sb_path)
	paths["side_by_side"] = sb_path

	return paths


@cli.command()
@click.option("--ref", "ref_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--test", "test_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out-dir", "out_dir", required=False, type=click.Path(file_okay=False), default='.')
@click.option("--base-name", "base_name", required=False, type=str, default='diff')
@click.option("--json", "as_json", is_flag=True, default=False, help="Output JSON")
def diff(ref_path: str, test_path: str, out_dir: str, base_name: str, as_json: bool):
	"""Compute metrics and create visual diffs between reference and test images.

	Saves visualizations to --out-dir and prints PSNR/SSIM. Use --json for machine-readable output.
	"""
	ref = load_image_as_numpy(ref_path)
	test = load_image_as_numpy(test_path)
	psnr = compute_psnr(ref, test)
	ssim = compute_ssim(ref, test)
	paths = _save_diff_images(ref, test, out_dir, base_name)
	result = {"psnr": float(psnr), "ssim": float(ssim), "paths": paths}
	if as_json:
		click.echo(json.dumps(result))
	else:
		click.echo(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
		click.echo("Saved:")
		for k, p in paths.items():
			click.echo(f" - {k}: {p}")


if __name__ == "__main__":
	cli()
