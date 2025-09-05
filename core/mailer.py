import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


OUTBOX_PATH = Path("data/outbox.jsonl")


def ensure_outbox():
	OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
	if not OUTBOX_PATH.exists():
		OUTBOX_PATH.touch()


def _json_default(o):
	if isinstance(o, (np.integer,)):
		return int(o)
	if isinstance(o, (np.floating,)):
		return float(o)
	if isinstance(o, (np.ndarray,)):
		return o.tolist()
	return str(o)


def send_email_simulated(subject: str, body: str, to: str = "stakeholders@example.com", meta: Dict[str, Any] = None) -> None:
	"""Append a simulated email to a JSONL outbox file for UI rendering."""
	ensure_outbox()
	record = {
		"to": to,
		"subject": subject,
		"body": body,
		"meta": meta or {}
	}
	with OUTBOX_PATH.open("a") as f:
		f.write(json.dumps(record, default=_json_default) + "\n")


def read_outbox(limit: int = 100) -> List[Dict[str, Any]]:
	ensure_outbox()
	rows: List[Dict[str, Any]] = []
	with OUTBOX_PATH.open() as f:
		for line in f.readlines()[-limit:]:
			try:
				rows.append(json.loads(line))
			except Exception:
				continue
	return rows


