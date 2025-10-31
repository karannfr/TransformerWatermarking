import hashlib
from typing import List, Tuple, Any

import numpy as np


def _seed_from_key(key: str) -> int:
	return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)


def keyed_permutation(items: List[Any], key: str) -> List[Any]:
	seed = _seed_from_key(key)
	rng = np.random.default_rng(seed)
	idx = np.arange(len(items))
	rng.shuffle(idx)
	return [items[i] for i in idx]
