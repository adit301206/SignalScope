# Project Metadata

## Evidence snapshot

| Item | Repository evidence |
|---|---|
| Binary architecture | EfficientNet-B0 with a two-output linear classifier |
| Binary checkpoint tracked | `src/models/best_efficientnet_b0.pth` |
| Mixed checkpoint path used by scripts | `model/best_efficientnet_b0_mixed.pth` (not tracked) |
| Attribution checkpoint path | `model/generator_attribution_30k.pth` (not tracked) |
| Recorded binary cross-generator evaluation | 600-image Defactify run; see `reports/mixed_results.md` |
| Recorded attribution evaluation | 600-image Defactify run; see `reports/attribution/evaluation_30k.txt` |
| Training date/version/commit for tracked checkpoint | Not recorded |
| Calibration artifact | Not implemented / not recorded |

The earlier metadata table labelled a CIFAKE-style/organizer dataset and a 2026-09-14 training date as the current model snapshot. Neither is tied to the tracked checkpoint by a committed manifest, history, or commit hash, so it is not presented as verified metadata.
