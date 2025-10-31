import argparse
import os
import zipfile

import requests
from tqdm import tqdm

COCO_TRAIN_2017 = "http://images.cocodataset.org/zips/train2017.zip"


def download(url: str, path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
	r = requests.get(url, stream=True)
	total = int(r.headers.get("content-length", 0))
	with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
		for chunk in r.iter_content(chunk_size=8192):
			if chunk:
				f.write(chunk)
				pbar.update(len(chunk))


def extract(zip_path: str, target_dir: str):
	with zipfile.ZipFile(zip_path, "r") as z:
		z.extractall(target_dir)


def main(target: str):
	os.makedirs(target, exist_ok=True)
	zip_path = os.path.join(target, "train2017.zip")
	if not os.path.exists(zip_path):
		print("Downloading COCO train2017...")
		download(COCO_TRAIN_2017, zip_path)
	else:
		print("Zip already present.")
	print("Extracting...")
	extract(zip_path, target)
	print("Done. Images under:", os.path.join(target, "train2017"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", type=str, required=True)
	args = parser.parse_args()
	main(args.target)
