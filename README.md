# Mustaff Models

Pre-trained model checkpoints for **[Mustaff](https://github.com/ChidcGithub/Mustaff)** — an ML-powered
automatic chart generator for 4-key osu!mania-style rhythm games.

This repository hosts versioned model weights (via GitHub Releases), evaluation metrics, and the
model manifest. Training data and code live elsewhere (see [Training Data](#training-data) and
[Reproducing](#reproducing)).

## Model Zoo

| Version | Component | Architecture | Params | Headline metric | Status |
|---|---|---|---|---|---|
| **v3.1** | placement | Beat-grid Transformer + diagnostic head (BPM/stars aux loss), EMA | 2.70M | **slot F1 0.671**, count ratio 1.11 | **current** |
| **v3.1** | selection | Autoregressive LSTM, 256 combos, **scheduled sampling** | 5.84M | combo acc 0.433 (TF) — generation hold ratio now stable at 10–12% per re-roll | **current** |
| v3.0 | placement | Beat-grid Transformer encoder, 48 slots/beat | 2.70M | slot F1 0.639 (±2 slots ≈ ±17 ms) | superseded |
| v3.0 | selection | Autoregressive LSTM over 256 column-state combos | 5.84M | combo acc 0.442 (prior 0.141) | superseded |
| v2 | placement | Frame-level Conv1d + FiLM + BiLSTM, multi-task (note/hold/len) | 1.22M | note F1 0.680 (±46 ms), hold F1 0.135 | legacy |
| v2 | selection | ContextCNN + LSTM, per-note 4-class | 0.71M | col acc 0.549 (prior 0.254) | legacy |

> v2 checkpoints are kept for reference and fallback. The runtime automatically prefers v3 when
> both generations are present (see `training/generate.py: v3_available`).

## Changelog

### v3.1 (2026-09-26)

- **Scheduled sampling** for selection (ramp to 25% self-fed history): fixes exposure bias that
  made long-note styling bistable at generation time (hold ratio swung 0% ↔ 66% between runs;
  now stable at 10–12% per re-roll)
- **Diagnostic head** on placement (regresses BPM/stars from pooled hidden state, λ = 0.3):
  forces the trunk to encode tempo/difficulty; slot F1 0.639 → 0.671, event count ratio
  1.52 → 1.11
- EMA weights (decay 0.999) used for evaluation and checkpoints
- Training throughput ~2.5× via bf16 autocast + 4-worker pinned DataLoaders + LR warmup

## What makes v3 different

v3 replaces 23 ms frame-level prediction with a **beat-quantized representation** (inspired by
DDC / ITGPT):

- **Placement** operates on a beat grid: each beat is divided into 48 slots, and the model
  predicts a multi-label "event" vector (note heads *and* long-note tails are both events).
  All generated notes land exactly on musical subdivisions (down to 1/64 notes) by construction.
- **Selection** assigns each event one of 4⁴ = 256 column-state combinations
  (`none / note / hold-head / hold-tail` per column). Chords and long notes are native outputs,
  not post-processing. Generation uses nucleus sampling with an n-gram repetition penalty
  (window 20, lengths 4–8), and long notes are decided from per-column marginal probabilities
  with structural constraints (a hold head requires a valid tail site within 2 beats).
- **Audio features** (82 channels @ ~43 fps): 64 log-mel, 12 chroma, percussive + broadband
  onset envelopes, RMS, beat phase, bar phase, BPM. Beat grids come from chart timing points
  during training and from librosa beat tracking at inference.
- **Difficulty conditioning** via FiLM (BPM, star rating). At inference the placement threshold
  is density-adaptive: it is binary-searched so that output density matches the requested star
  level (target NPS = 1.5 × stars).

## Training Data

- **2,912 beatmap sets** (osu!mania 4K, ranked / qualified / loved), mirrored from sayobot.cn
- **8,947 charts** after filtering (min 50 notes, 4K only), split 2,329 / 291 / 292
  (train / val / test, by set, seed 42)
- Difficulty labels are **NPS-based proxies** (notes-per-second × 1.5, clipped to [0.5, 12]),
  not official star ratings
- Column-mirror augmentation (24 permutations) is applied to selection training

## Usage

Download the four `*.pt` files from the latest
[Release](https://github.com/ChidcGithub/mustaff-models/releases) and place them in
`Mustaff/training/output/`. The GUI/CLI will detect them automatically (`生成方式 → ML 模型`
or `--ml`).

```
training/output/
├── placement_v3_best.pt    # v3 placement (beat-grid transformer)
├── selection_v3_best.pt    # v3 selection (256-combo LSTM)
├── placement_best.pt       # v2 fallback (optional)
└── selection_best.pt       # v2 fallback (optional)
```

## Reproducing

Training code lives in the main repository under `training/`.

```powershell
# 1. Collect beatmap sets (requires network; respects rate limits)
python -m training.collect_sets --max-sets 5000

# 2. Build features + labels (multi-process)
python -m training.build_dataset

# 3. Train (CUDA recommended; CPU works but is ~4x slower)
python -m training.train_placement_v3 --epochs 20 --batch 16
python -m training.train_selection_v3 --epochs 15 --batch 16

# 4. Smoke-test generation
python -m training.verify_gen 6.0
```

Reference hardware: RTX 4060 Laptop (8 GB), torch 2.6.0+cu124. v3 training takes ~45 min
(placement) + ~55 min (selection).

## Evaluation Notes

- **slot F1**: greedy match between predicted and ground-truth events in beat-slot space,
  ±2 slots tolerance (≈ ±17–25 ms depending on BPM). Stricter than v2's ±46 ms frame tolerance.
- **combo acc**: exact match over all 256 column-state combinations under teacher forcing.
- Long-note quality is currently limited by weak tail-release supervision; tails are mostly
  closed by structural rules (next note in column / 2-beat cap). Scheduled-sampling training
  is planned to address LN-style stability.

## Roadmap

- Scheduled sampling / DAgger for selection (fix exposure bias → stable long-note styling)
- Transformer decoder for selection (longer pattern context, ~500 events)
- Diagnostic network for BPM/difficulty alignment (ITGPT-style auxiliary loss)
- Pre-trained music encoder features (e.g. MERT) as a drop-in upgrade
- 6K / 7K key mode support

## License & Responsible Use

Model weights are derived from community-created osu!mania beatmaps. They are provided for
research and personal use. Generated charts are statistical remixes of human mapping styles —
credit original chart authors where appropriate, and do not upload generated maps as if they
were hand-made. Code snippets in this repo follow the main project's license.
