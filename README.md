# DSGen

DSGen is based on AeroGen. Branch `exp1` adds inference-only bounded instance attention; it uses the original model weights and introduces no trainable parameters.

## Experiment 1

| `--instance_attn` | Object branch | Self-attention |
| --- | --- | --- |
| `baseline` (default) | Original AeroGen, including padding tokens | Original |
| `padding` | Exclude padding tokens | Original |
| `cross` | Padding mask + instance token ownership | Original |
| `self` | Exclude padding tokens | Block other instances |
| `both` | Padding mask + instance token ownership | Block other instances |

Global text cross-attention and the existing `MaskCrossAttention` module are unchanged. For an object query, bounded self-attention permits keys from its own instance and the background. Background queries retain all keys. Background queries in object cross-attention can access all valid object tokens. An empty layout contributes zero object attention.

`bldm/instance_attention.py` maps the existing per-instance 64 x 64 masks to each attention resolution by area resampling. Each spatial query belongs to the instance with the largest coverage; ties use annotation order. A location with zero coverage is background. Padding masks are ignored when assigning ownership. This retains small masks during downsampling, but two tiny objects sharing one coarse cell can still compete for ownership.

Self-attention is bounded at 16 x 16 and 32 x 32 by default. To also try 64 x 64, pass `--self_resolutions 16 32 64`; this increases attention-mask memory. Object cross-attention is bounded at every transformer resolution. There is no timestep schedule or guidance/refinement/training change.

## Run inference

Run commands from the repository root in the `dsgen` environment. The original checkpoint, local CLIP and `dataset/category_embeddings.npy` are required. Defaults retain the original 512 x 512 output, 50 DDIM steps, eta 0.2, guidance 7.5 and two samples per layout. `--precision full` is the default.

```bash
# Original attention, with a fixed seed.
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python src/inference/inference.py \
  --instance_attn baseline --seed 42

# Run each ablation in a separate invocation, or pass all five modes together.
for mode in baseline padding cross self both; do
  CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python src/inference/inference.py \
    --instance_attn "$mode" --input_dir outputs/exp1/layouts \
    --output_dir "outputs/exp1/$mode" --seed 42 --steps 20 --num_samples 1
 done

# Recommended: load the checkpoint once and generate direct comparisons.
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python src/inference/inference.py \
  --instance_attn baseline padding cross self both \
  --input_dir outputs/exp1/layouts --output_dir outputs/exp1/results \
  --seed 42 --steps 20 --num_samples 1 --precision autocast
```

`--precision autocast` optionally uses CUDA mixed precision; use the same precision for every mode in a comparison. Each layout uses seed `seed + sorted_layout_index`. The runner reuses the same initial latent and restores CUDA RNG state before each mode, so DDIM eta noise is identical too. `PYTHONHASHSEED=0` also fixes the original dataset loader's ordering of multi-category prompt words across separate runs.

Outputs include `<mode>/<layout>_<sample>.png`, `comparisons/<layout>_<sample>.jpg`, `run.json`, and `stats.csv`. Comparison panels show a layout reference followed by the requested modes, with numbered box outlines. Statistics record seeds, instance counts, elapsed seconds per sampling batch and SHA-256 of the initial latent. They do not measure semantic merge/duplicate errors; those need visual review in this first experiment.

## Select real layouts and test

The small preparation script selects six layouts per group: adjacent vehicles/ships, dense mostly same-class objects, single objects, and sparse objects. Single-object cases are candidates for checking duplication; selection does not establish that the baseline actually duplicates them. It uses all annotations in each selected image and only selects images with at most the existing 20-object limit. It converts original DIOR rotated boxes to the existing normalized demo TXT format and records source XML paths in `manifest.json`.

```bash
python src/inference/prepare_exp1_layouts.py \
  --annotations /data2/cjy/data/DIOR-VOC/Annotations/Oriented_Bounding_Boxes \
  --output_dir outputs/exp1/layouts

python -m unittest src.test.test_instance_attention -v

CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 python src/inference/inference.py \
  --instance_attn baseline padding cross self both \
  --limit 1 --steps 5 --num_samples 1 --output_dir outputs/exp1_smoke
```

Tests check actual attention influence, object ownership, background behavior, padding, empty layouts, CFG batches, downsampling and default self-attention resolutions. The baseline transformer is checked bit for bit against the original `origin/main` implementation with identical parameters and inputs. Full checkpoint loading remains strict.

## Implementation

| File | Responsibility |
| --- | --- |
| `bldm/instance_attention.py` | Instance ownership, padding exclusion and cached masks per feature resolution |
| `ldm/modules/attention_dual.py` | Apply object/self masks before their respective softmax operations |
| `ldm/modules/diffusionmodules/openaimodel_bbox_v2.py` | Build and route the per-forward mask policy through U-Net transformers |
| `bldm/bldm_dual.py` | Pass the existing validity vector to U-Net |
| `src/inference/inference.py` | CLI, fixed-noise ablations, PNG outputs, comparisons and timing/noise records |
| `src/inference/prepare_exp1_layouts.py` | Select real DIOR layouts and record their source annotations |
| `src/test/test_instance_attention.py` | Baseline equivalence and attention-behavior checks |

## First run: 2026-10-28

Based on `origin/main` commit `98270ea`. Eleven attention tests passed, including bit-for-bit baseline transformer equivalence. A full-precision five-mode smoke test completed with strict checkpoint loading. The first ablation generated all 120 PNGs and 24 comparison rows from 24 DIOR layouts / 96 instances, using seed 42 + layout index, 20 DDIM steps, eta 0.2, guidance 7.5, one sample and CUDA autocast on an RTX 4090. Every layout's five modes have identical initial-noise hashes.

Artifacts are in `outputs/exp1/`: `layouts/manifest.json`, `layout_diagnostics.json`, `results/{mode}/`, `results/comparisons/`, four `results/overview_*.jpg` sheets, `results/run.json`, `results/stats.csv`, `results/summary.json`, and `REPORT.md`. These runtime artifacts remain outside Git.

The important negative result is that with one valid instance, these rules reduce to padding-only attention: foreground queries can already access themselves and all background, and only one object token remains. All six single-instance cases produced pixel-identical `padding`, `cross`, `self` and `both` outputs. The selected single-instance baselines did not show an obvious duplication failure, so this subset cannot establish a duplication improvement.

Visual review did not establish a consistent improvement. `adjacent_00033` and `adjacent_00103` still look like continuous ship bodies across the two touching annotation boxes in all five modes. The lower long box in the mixed-class `adjacent_00129` still contains multiple vehicle-like objects. Self-attention has a stronger effect on object appearance and surrounding scene structure, including the apron/jetways in `sparse_00020`, without a clear count benefit. Tiny-object misses were not exhaustively counted.

Cross differs little from padding: mean absolute RGB difference is 0.7371 on the 0-255 scale, versus 7.8992 for self and 7.9198 for both. These numbers measure image changes, not quality or instance error rates. Ownership diagnostics show 11 instances without their own cell at 8 x 8 and six at 16 x 16 (all six in `dense_00035`); all 96 retain an owned cell at 32 x 32 and 64 x 64.

A useful next test is to constrain object cross-attention only at finer resolutions, then repeat confirmed failure layouts with 50 steps and multiple seeds. A separate mechanism is needed to constrain multiple objects inside one instance region; the present inter-instance isolation cannot provide that for a truly single-instance input.
