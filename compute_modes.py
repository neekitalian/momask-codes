import json, numpy as np

data = json.load(open(r'C:\Users\maura\source\repos\momask-codes\.claude\worktrees\fervent-satoshi-4e02b9\semantic_spectrum_data\spectrum_scores.json'))
verbs = ['walk','dance','run','jump','spin','kick','wave','stand']
scores = {v: np.array([entry[v] for entry in data.values()]) for v in verbs}

N_QUANTILES = 10
BIN_SIZE = 0.05

print(f'Modes per verb per quantile (bin={BIN_SIZE}, Q={N_QUANTILES}):')
modes = {}
for v in verbs:
    s = scores[v]
    boundaries = np.quantile(s, np.linspace(0, 1, N_QUANTILES + 1))
    verb_modes = []
    for i in range(N_QUANTILES):
        lo, hi = boundaries[i], boundaries[i+1]
        mask = (s >= lo) & (s <= hi) if i == N_QUANTILES-1 else (s >= lo) & (s < hi)
        bucket = s[mask]
        binned = np.round(bucket / BIN_SIZE) * BIN_SIZE
        vals, counts = np.unique(np.round(binned, 4), return_counts=True)
        mode_val = float(vals[np.argmax(counts)])
        verb_modes.append(round(mode_val, 4))
    modes[v] = verb_modes
    print(f'  {v}: {verb_modes}')

print('\nQuantile boundaries per verb:')
for v in verbs:
    s = scores[v]
    boundaries = np.round(np.quantile(s, np.linspace(0, 1, N_QUANTILES + 1)), 4).tolist()
    print(f'  {v}: {boundaries}')
