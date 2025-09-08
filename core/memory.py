import json
from pathlib import Path
from typing import Dict, Any
import numpy as np


MEMORY_PATH = Path("data/memory.jsonl")


def _json_default(o):
	if isinstance(o, (np.integer,)):
		return int(o)
	if isinstance(o, (np.floating,)):
		return float(o)
	if isinstance(o, (np.ndarray,)):
		return o.tolist()
	return str(o)


def append_memory(event: Dict[str, Any]) -> None:
	MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
	with MEMORY_PATH.open("a") as f:
		f.write(json.dumps(event, default=_json_default) + "\n")



