# CEA Reviewer #2 pre-submission risk list (minimal-cost closure)

## Top-10 likely rejection risks (severity-ordered)
1. Strong `drift-proof` claim without formal definition and trigger/mitigation loop evidence.
2. Gating threshold plateau (1000/2000/4000 identical) without activation-level analysis.
3. Runtime reporting scope ambiguity (tracking-only vs end-to-end mixed or unspecified).
4. Novelty risk: perceived as engineering integration rather than reusable methodology.
5. Stress tests too narrow for aquaculture realism (drop/jitter only).
6. Cross-domain generalization evidence missing (single split only).
7. Drift trigger calibration not specified (false alarm vs miss trade-off absent).
8. Uncertainty/statistical protocol incomplete in key sections (especially runtime and sensitivity).
9. Data/code availability still placeholder-level for archival expectations.
10. Terminology inconsistency across sections (drift, gating score, CPU normalization).

## Minimal-cost remediation map (no large new dataset required)
| Risk | Minimal patch target | Low-cost addition |
|---|---|---|
| 1 | Title/Abstract/Methods/Conclusion | Replace with drift-aware wording + add formal drift scope paragraph |
| 2 | Results (gating section) | Add trigger rate, event length, score CDF TODO templates |
| 3 | Methods + Runtime section | Split protocol into A tracking-only / B end-to-end |
| 4 | Introduction contributions | Compress to C1/C2/C3 with evidence mapping |
| 5 | Methods stress tests | Add 3 degradations × 3 strengths template table |
| 6 | New cross-domain section | Add low/med/high-cost plans + metrics checklist |
| 7 | Drift subsection | Add quantile-based threshold calibration and persistence K |
| 8 | Runtime + sensitivity text | Add repeats/warm-up/statistics TODO protocol |
| 9 | Data/code section | Keep TODO DOI/link with explicit release scope |
| 10 | Nomenclature block | Add symbol table and normalized CPU definition |

## Rebuttal template (opinion -> response -> changes)
| Reviewer comment | Response | Manuscript changes |
|---|---|---|
| Drift claim too strong | We agree and replaced proof-like wording with drift-aware monitoring language. | Title, Abstract, Methods, Conclusion updated. |
| Drift not formalized | We added formal input/output drift definitions and a closed-loop algorithm. | Added subsection “Drift definition and closed-loop handling” and Algorithm block. |
| Threshold sweep not convincing | We added gating activation diagnostics and quantile-based threshold selection protocol. | Added subsection “Gating activation analysis”, Fig/Table TODO list. |
| Runtime protocol unclear | We split runtime scope into tracking-only and end-to-end and defined fields. | Added runtime protocol subsection and field-definition table. |
| Stress tests not realistic | We expanded degradations to blur/low-light/haze with parameter templates. | Added “Environment-aware stress tests” subsection and parameter table. |
| Generalization missing | We added a staged cross-domain template with transferable vs recalibrated settings. | Added “Cross-domain generalization” section with reporting checklist. |
| Contribution framing weak | We condensed contributions to three reusable methodological claims with evidence links. | Introduction contribution list rewritten as C1/C2/C3. |
| Statistical robustness concerns | We added minimal-repeat/statistical TODO protocol without inventing results. | Runtime and gating sections include explicit TODO experiment closures. |
| Availability unclear | We clarified release package scope and kept archival links as TODO placeholders. | Data/Code/Repro statements refined. |
| Terminology inconsistency | We added nomenclature and harmonized terms across sections. | Added Nomenclature/notation block and terminology cleanup. |
