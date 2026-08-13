"""Compute per-verb quantile modes from the actual synthetic training captions."""
import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TEXTS_DIR = Path(r"C:\Users\maura\source\repos\momask-reference\dataset\HumanML3D\texts")
VERBS     = ['walk', 'dance', 'run', 'jump', 'spin', 'kick', 'wave', 'stand']
N_Q       = 10
BIN_SIZE  = 0.01
FIXED_BIN = 0.1   # fixed-width bins matching the report charts

# Collect all verb scores from spec files
scores = defaultdict(list)
spec_files = list(TEXTS_DIR.glob("*_spec_*.txt"))
print(f"Reading {len(spec_files)} spec files ...")

for f in spec_files:
    line = f.read_text(encoding='utf-8').split('\n')[0]
    prompt_part = line.split('#')[0]   # only the prompt before the token field
    for match in re.finditer(r'\[(\w+):(0\.\d+)\]', prompt_part):  # require leading 0.
        verb, val = match.group(1), float(match.group(2))
        if verb in VERBS:
            scores[verb].append(val)

print(f"Scores collected per verb:")
for v in VERBS:
    print(f"  {v}: {len(scores[v])} values")

print(f"\nModes (fixed-width bins of {FIXED_BIN}, score bins 0-10%, 10-20%, ...):")
modes = {}
for v in VERBS:
    s = np.array(scores[v])
    verb_modes = []
    for i in range(N_Q):
        lo = round(i * FIXED_BIN, 2)
        hi = round((i + 1) * FIXED_BIN, 2)
        mask = (s >= lo) & (s < hi) if i < N_Q - 1 else (s >= lo) & (s <= hi)
        bucket = s[mask]
        if len(bucket) == 0:
            verb_modes.append(round(lo + FIXED_BIN / 2, 2))
            continue
        binned = np.round(bucket / BIN_SIZE) * BIN_SIZE
        vals, counts = np.unique(np.round(binned, 4), return_counts=True)
        mode_val = float(vals[np.argmax(counts)])
        verb_modes.append(round(mode_val, 4))
    modes[v] = verb_modes
    print(f"  '{v}': {verb_modes},")

Path("caption_modes.json").write_text(json.dumps(modes, indent=2))
print("\nSaved to caption_modes.json")
