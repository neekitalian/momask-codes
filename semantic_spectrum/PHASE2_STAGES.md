# Phase 2 Stages 2.2 to 2.4

Pure-kinematics modules that extend the zone-blend pipeline. No torch and no GPU are
needed to run or test them (`pytest tests/test_phase2_stages.py`). The one GPU step (the
model retrain in Stage 2.3) is called out explicitly below.

## Stage 2.2 - z estimation and integration (`estimate_z.py`)

Replaces the hand-written `[walk:0.91][dance:0.09]` label with one estimated from the
performer's motion. The six semantic verbs are split into a base (walk, run, stand) and a
style (dance, spin, jump) group; z is the style fraction on the dominant axis:

    z = style_score / (style_score + base_score)

```python
from semantic_spectrum.estimate_z import estimate_prompt
prompt = estimate_prompt("input_videos/momask_input.npz", text="A person walks.")
# -> "[walk:0.78][dance:0.22] A person walks."
```

Integration point: in `scripts/video_to_spectrum.py`, swap the fixed walk term for
`estimate_prompt(npz)`. `gamma` warps z for calibration (Stage 2.3 fits it).

## Stage 2.3 - binning and feedback calibration (`calibrate.py`)

Snaps z onto the model's discrete mode grid, and turns the Round 2 perceptual study into a
per-genre blending alpha. Allocation level (20 / 40 / 60) is the study's style-strength knob.

```python
from semantic_spectrum.calibrate import calibrate_alpha, write_calibration
cal = calibrate_alpha("semantic_spectrum_data/round2_summary.csv", objective="balanced")
write_calibration("semantic_spectrum_data/alpha_by_genre.json", cal)
```

Result on the Round 2 data (balanced objective). Identity falls and expressivity rises with
allocation, and 40 is the trade-off point for every genre:

| genre   | 20 (id / expr) | 40 (id / expr) | 60 (id / expr) | chosen alpha |
|---------|----------------|----------------|----------------|--------------|
| ballet  | 6.21 / 3.77    | 5.61 / 4.44    | 5.20 / 4.74    | 0.40         |
| hip_hop | 6.16 / 4.04    | 5.84 / 4.35    | 4.97 / 4.84    | 0.40         |
| jazz    | 6.36 / 3.80    | 5.86 / 4.26    | 5.25 / 4.59    | 0.40         |

`objective` also supports `identity` and `expressivity` (the latter maximises expressivity
subject to an identity floor). Retrain (GPU, out of scope here): feed the rebinned labels to
`train_t2m_transformer.py` / `train_res_transformer.py`.

## Stage 2.4 - generalizability testing (`evaluate.py`)

Scores (original, output) joint pairs on four metrics and aggregates across genre, verb and
performer. A run generalizes when the metrics stay high and steady on every slice.

```python
from semantic_spectrum.evaluate import evaluate_matrix, generalizability_report, format_report
metrics = evaluate_matrix(records)   # records: {original, output, genre, verb, performer}
print(format_report(generalizability_report(metrics)))
```

Metrics: `identity_preservation`, `style_strength`, `smoothness` (all higher is better) and
`foot_skate` (lower is better). `cross_performer_consistency` reports how steady each metric
stays as the performer changes.
