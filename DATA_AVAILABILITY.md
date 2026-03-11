# Data Availability

This repository contains code, experiment scripts, and derived paper artifacts. It should not be read as a blanket statement that all local data folders or model files in a private working copy are publicly redistributable.

## What Can Usually Be Released

The following items are generally suitable for a paper release package, subject to any third-party license terms already attached to the repository:

- source code under the repository
- experiment scripts under `scripts/`
- frozen configuration files such as `results/main_val/run_config.json`
- derived tables, plots, and manifest records under `results/main_val/`
- documentation files such as `REPRODUCIBILITY.md` and `manifest.template.json`

## What Should Not Be Claimed Public Without Explicit Permission

The following categories may exist locally but should not be described as public unless the release archive actually contains them and redistribution is allowed:

- raw farm videos
- prepared MOT-style exports derived from restricted raw videos
- locally stored checkpoints such as `runs/traj_encoder/traj_encoder.pt`
- local detector weight files such as `yolov8n.pt` or `yolo26n.pt`
- archived intermediate result bundles produced during private drafting

## Honest Release Language

If the public package includes only code and derived results, an accurate statement is:

> Code, experiment scripts, frozen run configurations, and derived paper artifacts are released. Raw videos, prepared benchmark exports, and model checkpoints are released only when redistribution rights are available; otherwise they remain unavailable in the public package.

If a reviewer or editor receives a larger private bundle than the public archive, that difference should be stated explicitly rather than implied away.

## Practical Reproduction Consequence

- If `data/mft25_mot_full` and `runs/traj_encoder/traj_encoder.pt` are available locally, the core paper figures can be regenerated from code.
- If those assets are not released, the repository can still rerender many paper-facing figures from released CSV outputs, but that is a partial artifact reproduction, not a full end-to-end rerun.
- The optional drift operating-point scan additionally depends on `results/brackishmot_drift_eval_timeline.csv`. If that file is absent from a release, the drift scan should be marked unavailable.
