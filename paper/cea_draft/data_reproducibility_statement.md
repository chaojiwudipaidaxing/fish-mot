# Data Availability and Reproducibility Statement

This manuscript uses the active authority results tree:
`results/main_val`.

- Manifest: `release/manifest.json`
  - git_commit: `d03740f78ed83caa11f727f9f7f060491434b199`
  - input entries: 10 total (`sha256` on 9 entries; `run_config.json` is recorded by path-only policy to avoid a circular nested hash)
  - hashed outputs: 18
- Locked experiment config: `run_config.json`
- Reproduction entrypoint: `release/reproduce.bat`

The bundle includes derived metrics, editable tables (CSV/TEX), and figures used by the draft.
If raw videos are subject to licensing/privacy constraints, the minimum reproducible package
(derived annotations + scripts + manifest hashes) should be released publicly.
