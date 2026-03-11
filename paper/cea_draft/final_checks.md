# Final Checks (CEA submission gate)

## Gate summary

| Gate | Status | Evidence |
|---|---|---|
| 6.1 Authority/Sub package consistency hash | PASS | `scripts/sync_submission_package.py` prints identical normalized hash (`c7fc2f...5670`) |
| 6.2 Forbidden-claim and placeholder scan | PASS | project text scan returns no disallowed claim terms and no placeholder tokens |
| 6.3 Figure/table reference completeness | PASS | `scripts/sync_submission_package.py` and `scripts/build_em_flat_package.py` report all referenced `figs/*` and `tables/*` files as PASS |
| 6.4 EM zip flatness | PASS | `CEA_EM_flat.zip` contains `bad_entries=0` for `/` or `\\` in zip entry names |
| 6.5 Numeric traceability coverage | PASS | prose-number audit over Abstract/Results/Discussion reports `missing=0` against `sanity_check.md` |
| Figure/Table production style gate | PASS | all submission figures re-rendered at ~500 DPI; table tex uses `booktabs` + `siunitx` uncertainty format; no plain-text plus/minus or code-style field labels in manuscript tables |

## CEA compliance blocks in manuscript

| Required block in `main.tex` | Status |
|---|---|
| Data availability statement | PASS |
| Declaration of competing interests | PASS |
| Funding statement | PASS |
| Author contributions (CRediT) | PASS |
| AI declaration (if AI used) | PASS (`Declaration of Generative AI and AI-assisted technologies in the manuscript preparation process`) |
| Appendix: Release package (DOI/anonymous link placeholder) | PASS |

## Highlights and graphical abstract checks

| Item | Status | Evidence |
|---|---|---|
| Highlights file exists in submission package | PASS | `paper/cea_draft/submission_package/highlights.txt` |
| Highlights count/length | PASS | 5 lines, max 74 chars |
| Graphical abstract exists | PASS | `paper/cea_draft/submission_package/graphical_abstract.png` |
| Graphical abstract size | PASS | 4044x1652 px |
| Artwork provenance | PASS | artwork non-AI generated; figures/graphical abstract come from experiment assets and manual drawing |

## Package outputs

| Package | Status | Path |
|---|---|---|
| Overleaf package | PASS | `paper/cea_draft/submission_package/source.zip` |
| Elsevier EM flat package | PASS | `paper/cea_draft/submission_package/CEA_EM_flat.zip` |

## Commands run (latest)

```powershell
.\.venv\Scripts\python.exe scripts\make_paper_assets.py ... --out-dir paper\cea_draft\_regen_assets ...
.\.venv\Scripts\python.exe scripts\run_strong_baselines.py --render-only ...
.\.venv\Scripts\python.exe scripts\<gating-sensitivity-render>.py --render-only ...
.\.venv\Scripts\python.exe scripts\run_degradation_grid.py --render-only ...
.\.venv\Scripts\python.exe scripts\eval_stratified.py --plot-only ...
.\.venv\Scripts\python.exe scripts\profile_runtime.py --plot-only ...
.\.venv\Scripts\python.exe scripts\sync_submission_package.py --draft-dir paper/cea_draft --submission-dir paper/cea_draft/submission_package --zip-name source.zip
.\.venv\Scripts\python.exe scripts\build_em_flat_package.py --submission-dir paper/cea_draft/submission_package --flat-dir-name em_flat --zip-name CEA_EM_flat.zip
```

Key outputs:
- `sync_submission_package.py`: all references PASS, hash match PASS, highlights PASS, `source.zip` rebuilt.
- `build_em_flat_package.py`: all references PASS, `CEA_EM_flat.zip` built, no subdirectories.
- formatting integrity script: PASS (no numeric drift, no forbidden manuscript tokens, 8/8 figures present at ~500 DPI).

Additional gate output snapshot:

```text
C6.1 PASS c7fc2fa003a4a3d4f7e829148206ceb242ccb303da97ecf96600387283d55670
C6.2 PASS hits=0
C6.3 PASS missing=0
C6.4 PASS bad_entries=0
C6.5 PASS missing_numbers=0
FIGSTYLE PASS dpi~=500 table_style=booktabs+siunitx numeric_drift=0
```

## Remaining manual items before actual submission

1) Replace CRediT placeholders (`[Author 1]`, `[Author 2]`) with real names.  
2) Confirm final data-sharing language with project/license owner.  
3) Upload declarations-tool `.docx` statement in Editorial Manager.  
4) Compile PDF on Overleaf from the refreshed `source.zip` and do final visual proofreading.
