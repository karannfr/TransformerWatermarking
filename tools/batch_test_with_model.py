#!/usr/bin/env python3
"""Batch test tool (with importance model).

For each image in --input-dir (up to --limit), embed a random payload using a random key and the
importance model, then immediately extract and validate the extracted payload equals the original.
Also compute PSNR/SSIM between original and watermarked images and write results to CSV/JSON.

Usage:
python tools\batch_test_with_model.py --input-dir ./images --output-dir ./results_with_model --model ../best_model.pth --ecc-symbols 64
"""
from __future__ import annotations
import argparse
import base64
import csv
import json
import os
import random
import string
import subprocess
import time
from pathlib import Path
from typing import Tuple
from PIL import Image

DEFAULT_MESSAGE_BYTES = 28  # 28 bytes == 224 bits


def random_key(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits + "_-@"
    return "".join(random.choice(alphabet) for _ in range(length))


def run_cmd(cmd, timeout=180) -> Tuple[int, str, str, float]:
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dur = time.time() - start
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip(), dur


def parse_quality_output(out: str) -> Tuple[float | None, float | None]:
    psnr = None
    ssim = None
    try:
        if "PSNR:" in out and "SSIM:" in out:
            left, _ = out.split("SSIM:")
            psnr_part = left.split("PSNR:")[-1].strip().split()[0]
            ssim_part = out.split("SSIM:")[-1].strip().split()[0]
            psnr = float(psnr_part)
            ssim = float(ssim_part)
    except Exception:
        pass
    return psnr, ssim


def single_test_with_model(image_path: Path, out_dir: Path, ecc_symbols: int, message_bytes: int, model_path: str) -> dict:
    results = {}
    fname = image_path.name
    key = random_key()
    payload = os.urandom(message_bytes)
    payload_b64 = base64.b64encode(payload).decode("ascii")

    wm_path = out_dir / f"{fname}.wm.png"

    # Embed with model
    embed_cmd = [
        "python", "-m", "watermark_cli", "embed",
        "--input", str(image_path),
        "--output", str(wm_path),
        "--message-b64", payload_b64,
        "--key", key,
        "--ecc-symbols", str(ecc_symbols),
        "--use-importance",
        "--model", str(model_path)
    ]
    rc, out, err, t_embed = run_cmd(embed_cmd)
    results.update({
        "filename": fname,
        "key": key,
        "ecc_symbols": ecc_symbols,
        "message_bytes": message_bytes,
        "model": model_path,
        "embed_returncode": rc,
        "embed_stdout": out,
        "embed_stderr": err,
        "embed_time_s": round(t_embed, 3)
    })

    # Quality original -> watermarked
    qcmd = ["python", "-m", "watermark_cli", "quality", "--ref", str(image_path), "--test", str(wm_path)]
    rcq, outq, errq, tq = run_cmd(qcmd)
    psnr_wm, ssim_wm = parse_quality_output(outq)
    results.update({"psnr_wm": psnr_wm, "ssim_wm": ssim_wm})

    # Extract from watermarked (original)
    extract_cmd = [
        "python", "-m", "watermark_cli", "extract",
        "--input", str(wm_path),
        "--length", str(message_bytes * 8),
        "--key", key,
        "--ecc-symbols", str(ecc_symbols),
        "--use-importance",
        "--model", str(model_path),
        "--as-b64"
    ]
    rce, oute, erre, te = run_cmd(extract_cmd)
    results.update({
        "extract_returncode": rce,
        "extract_stdout": oute,
        "extract_stderr": erre,
        "extract_time_s": round(te, 3)
    })

    success = False
    if rce == 0 and oute and "decode failed" not in oute.lower():
        decoded_b64 = oute.strip().splitlines()[-1].strip()
        success = (decoded_b64 == payload_b64)
    results["success"] = success

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True, help="Path to importance model checkpoint")
    p.add_argument("--ecc-symbols", type=int, default=64)
    p.add_argument("--message-bytes", type=int, default=DEFAULT_MESSAGE_BYTES)
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in inp.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])[:args.limit]
    all_results = []
    for img in images:
        print("Testing", img.name)
        r = single_test_with_model(img, out, args.ecc_symbols, args.message_bytes, args.model)
        all_results.append(r)
        # write incremental JSON and CSV
        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        keys = list(all_results[0].keys())
        with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

    succ = sum(1 for r in all_results if r.get("success"))
    print(f"Done. {succ}/{len(all_results)} recovered from watermarked files.")


if __name__ == "__main__":
    main()
