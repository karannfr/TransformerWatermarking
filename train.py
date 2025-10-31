import argparse
import os
from glob import glob

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from watermark_cli.models.transformer import ViTImportance, sobel_pseudolabels


class ImageFolderDataset(Dataset):
	def __init__(self, root: str, exts=(".jpg", ".jpeg", ".png"), size: int = 256):
		self.paths = []
		for e in exts:
			self.paths.extend(glob(os.path.join(root, f"**/*{e}"), recursive=True))
		self.size = size

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx: int):
		p = self.paths[idx]
		img = Image.open(p).convert("RGB").resize((self.size, self.size))
		arr = np.array(img).astype(np.float32) / 255.0
		arr = torch.from_numpy(arr).permute(2, 0, 1)
		return arr


def train(args):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ds = ImageFolderDataset(args.data, size=args.size)
	dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

	model = ViTImportance(patch_size=args.patch_size, embed_dim=args.embed_dim, depth=args.depth, heads=args.heads).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
	crit = nn.L1Loss()

	for epoch in range(args.epochs):
		model.train()
		pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
		for batch in pbar:
			batch = batch.to(device)
			with torch.no_grad():
				y = sobel_pseudolabels(batch)
			pred = model(batch)
			loss = crit(pred, y)
			opt.zero_grad()
			loss.backward()
			opt.step()
			pbar.set_postfix({"loss": float(loss.item())})

	os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
	model.save_checkpoint(args.out)
	print(f"saved: {args.out}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, required=True, help="Path to COCO images root (e.g., data/coco2017/train2017)")
	parser.add_argument("--out", type=str, default="./checkpoints/vit_sobel.pth")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--size", type=int, default=256)
	parser.add_argument("--patch-size", type=int, default=8)
	parser.add_argument("--embed-dim", type=int, default=192)
	parser.add_argument("--depth", type=int, default=4)
	parser.add_argument("--heads", type=int, default=4)
	parser.add_argument("--lr", type=float, default=3e-4)
	args = parser.parse_args()
	train(args)
