from typing import Optional

import numpy as np
from reedsolo import RSCodec, ReedSolomonError


def ecc_encode_bits(bits: np.ndarray, rs_parity_symbols: int = 64) -> np.ndarray:
	"""Encode bit array using Reed-Solomon over bytes.
	- Pack bits to bytes, encode, then unpack to bits.
	"""
	bits = bits.astype(np.uint8)
	n = (bits.size + 7) // 8
	pad = n * 8 - bits.size
	if pad > 0:
		bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
	payload_bytes = np.packbits(bits)
	rsc = RSCodec(rs_parity_symbols)
	coded = rsc.encode(bytes(payload_bytes))
	coded_bits = np.unpackbits(np.frombuffer(coded, dtype=np.uint8)).astype(np.uint8)
	return coded_bits


def ecc_decode_bits(coded_bits: np.ndarray, expected_payload_bits: int, rs_parity_symbols: int = 64) -> Optional[np.ndarray]:
	"""Decode bit array using Reed-Solomon over bytes.
	Returns None if decode fails. On success, trims to expected_payload_bits.
	"""
	coded_bits = coded_bits.astype(np.uint8)
	n = (coded_bits.size // 8) * 8
	coded_bits = coded_bits[:n]
	coded_bytes = np.packbits(coded_bits)
	try:
		rsc = RSCodec(rs_parity_symbols)
		decoded = rsc.decode(bytes(coded_bytes))[0]
		decoded_bits = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8)).astype(np.uint8)
		return decoded_bits[:expected_payload_bits]
	except ReedSolomonError:
		return None
