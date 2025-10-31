## Watermark CLI: Transformer + ECC + LSB2

A Python CLI tool to embed and extract invisible watermarks in images using 2-bit LSB embedding (LSB2) and Reed-Solomon ECC. A lightweight Vision Transformer is provided for research; current CLI defaults to deterministic selection using a secret key for robust round-trips.

### Install (Windows PowerShell)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Quickstart

```powershell
# Embed a message
python -m watermark_cli embed --input .\input.jpg --output .\watermarked.png --message "hello world" --key "mysecret"

# Extract it back (11 chars → 88 bits)
python -m watermark_cli extract --input .\watermarked.png --key "mysecret" --length 88 --as-text --print

# Quality (vs original)
python -m watermark_cli quality --ref .\input.jpg --test .\watermarked.png
```

---

## CLI User Guide

### Commands

- Embed watermark:

```powershell
python -m watermark_cli embed --input <in> --output <out> --message "text" --key "secret" [--channels rgb] [--ecc-symbols 64] [--use-importance] [--model <ckpt>] [--json]
```

- Extract watermark:

```powershell
python -m watermark_cli extract --input <in> --key "secret" --length <bits> [--as-b64] [--channels rgb] [--ecc-symbols 64] [--use-importance] [--model <ckpt>] [--json]
```

- Quality metrics:

```powershell
python -m watermark_cli quality --ref <ref> --test <test> [--json]
```

### Required arguments

- embed
  - `--input`: path to source image
  - `--output`: path to save watermarked image (PNG recommended)
  - one of: `--message "text"` or `--message-b64 "base64payload"`
  - `--key "secret"`: secret for deterministic position selection
- extract
  - `--input`: watermarked image
  - `--key "secret"`: must match the embed key
  - `--length <bits>`: original payload length in bits (before ECC)

### Common options

- `--channels`: which channels to use: `rgb|r|g|b` (default: `rgb`)
- `--ecc-symbols`: Reed–Solomon parity bytes (default: `64`). Use the same value on extract if you changed it from default during embed.
- `--use-importance`: use transformer-generated importance map to guide selection
- `--model`: path to transformer checkpoint (default: `./checkpoints/vit_sobel.pth`)
- `--as-b64/--as-text`: controls decoded output format on extract (default: `--as-text`)
- `--json`: outputs structured JSON instead of plain text

### JSON output examples

- Embed:

```powershell
python -m watermark_cli embed --input .\in.jpg --output .\out.png --message "hello" --key "k" --json
# {"status":"ok","output":".\\out.png","bits":<coded_bits>}
```

- Extract:

```powershell
python -m watermark_cli extract --input .\out.png --key "k" --length 40 --as-text --json
# {"status":"ok","message":"hello"}
```

- Quality:

```powershell
python -m watermark_cli quality --ref .\in.jpg --test .\out.png --json
# {"psnr": 42.1, "ssim": 0.9967}
```

### Workflows

1. Text payload round‑trip

```powershell
python -m watermark_cli embed --input .\input.jpg --output .\watermarked.png --message "hello world" --key "mysecret"
# 11 chars × 8 = 88 bits
python -m watermark_cli extract --input .\watermarked.png --key "mysecret" --length 88 --as-text --print
```

2. Base64 payloads

```powershell
# Embed base64 payload
python -m watermark_cli embed --input .\input.jpg --output .\watermarked.png --message-b64 "SGVsbG8gV29ybGQ=" --key "mysecret"
# For extraction, use the original decoded-bytes length × 8 as --length
python -m watermark_cli extract --input .\watermarked.png --key "mysecret" --length 88 --as-b64 --print
```

3. Transformer-guided selection (optional)

```powershell
# Ensure a trained checkpoint is available
# e.g., .\checkpoints\vit_sobel.pth

# Embed with transformer importance
python -m watermark_cli embed --input .\input.jpg --output .\watermarked.png --message "hello world" --key "mysecret" --use-importance --model .\checkpoints\vit_sobel.pth --json

# Extract with the SAME settings
python -m watermark_cli extract --input .\watermarked.png --key "mysecret" --length 88 --use-importance --model .\checkpoints\vit_sobel.pth --as-text --json
```

4. Quality check

```powershell
python -m watermark_cli quality --ref .\input.jpg --test .\watermarked.png
# PSNR: 40.00 dB, SSIM: 0.9950
```

### ECC and bit-length

- Raw payload bits:
  - Text: `len(text.encode('utf-8')) * 8`
  - Base64: `len(base64.b64decode(s)) * 8`
- ECC parity is handled internally during embed. On extract, provide the original payload bit-length via `--length`. If you changed `--ecc-symbols` at embed time, pass the same value at extract time.

### Channels and capacity

- LSB2 stores 2 bits per selected pixel‑channel.
- With `--channels rgb`, each pixel contributes 3 positions → 6 bits per pixel. Using `r` only yields 2 bits per pixel.
- The tool automatically selects enough positions for the ECC‑coded payload length.

### Determinism and keys

- The selection of positions is deterministic given: `--key`, `--channels`, `--ecc-symbols`, and (if enabled) the transformer importance map and checkpoint.
- To extract successfully, these must match the values used during embedding.

### File format guidance

- Prefer saving watermarked outputs as PNG to avoid destructive JPEG recompression.
- You can embed into a JPEG input and output PNG.

### Transformer (training optional)

```powershell
# COCO 2017 train images
python data\coco_download.py --target .\data\coco2017
python train.py --data .\data\coco2017\train2017 --epochs 5 --batch-size 16 --out .\checkpoints\vit_sobel.pth

# Or any folder of images
python train.py --data .\data\kaggle\your_images --epochs 5 --batch-size 16 --out .\checkpoints\vit_sobel.pth
```

### Troubleshooting

- "decode failed":
  - Ensure `--key`, `--channels`, `--ecc-symbols`, and (if used) `--use-importance` + `--model` match embed settings.
  - Confirm `--length` equals the original payload bits.
  - Use PNG for the output image to avoid recompression artifacts.
- Torch install issues:
  - Prefer Python 3.12, or upgrade pip and use the torch version available for your interpreter/OS.

### Notes

- This implementation is for research/education; it is not a security product.
- Robustness depends on image content and any post‑processing applied after embedding.
