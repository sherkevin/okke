# CHORD Rebuttal Artifact Index - 2026-05-30

## Versioning rule

- Before editing the 5-page scientific master, first copy the latest timestamped `author_response_min_diff_expected_YYYYMMDD_HHMM.tex` / `.pdf` to a new timestamped basename using the current date and minute, then edit the new `.tex`. This preserves review history and avoids regenerating the response from scratch.
- The active 5-page scientific master should always be the latest timestamped `author_response_min_diff_expected_YYYYMMDD_HHMM.pdf`, while the one-page upload-target draft remains a distilled derivative.
- Under the latest PC Chairs email, the only official rebuttal deliverable is one strict one-page PDF. The 5-page timestamped master is internal only; Official Comments must not be used for reviewer replies or overflow content.

## Current canonical deliverable

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1541.pdf`
  - Rich rebuttal response PDF with the expected-result tables, remaining-gap closure tables, camera-ready detail closure, and the revised Figure 2 sketch embedded.
  - Status: current best expected-result scientific master PDF draft, not the uploadable ACM MM rebuttal. The prose now argues from the expected-result pattern; superscript `E` marks the values to replace once official result values arrive. The one-page upload-target draft should be distilled from this master.
  - Format boundary: the authenticated ACM MM OpenReview task for Submission 8826 is `Rebuttal PDF` and requires a single-page PDF. This 5-page master must be distilled before upload.
  - Current verification: 5 pages, 340525 bytes, modified 2026-05-30 15:42:37, SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected_20260530_1541.tex`
  - Source for the current canonical PDF.
  - SHA256 `67FEFD54814E424D4A9CDFAA91B83390E5F4A2F3DBED5B73AF816C6EB6292349`.

## One-page upload-target draft

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_upload_target_expected_20260530.pdf`
  - One-page expected-result upload-target draft distilled from the 5-page Weak-Accept master. It preserves the key reviewer-conversion evidence in a compact evidence matrix and keeps an explicit internal expected-placeholder note.
  - Status: internal expected-results draft only; before real submission, replace expected values, delete the internal placeholder note and `E` markers, and re-check the active OpenReview form.
  - Format rule from PC Chairs email: all rebuttal content, references, and supporting material must fit on this single page; do not rely on Official Comment for any reviewer-facing content.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_upload_target_expected_20260530.tex`
  - Source for the one-page upload-target draft.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_upload_target_expected_20260530_preview.png`
  - Rendered visual check for the current one-page upload-target draft.
- Verification snapshot:
  - `pdfinfo author_response_upload_target_expected_20260530.pdf` reports 1 page, letter page size, 159558 bytes.
  - LaTeX log scan reports no `LaTeX Error`, `Fatal error`, `Emergency stop`, `Overfull`, or `LaTeX Warning` lines.
  - `pdftotext -layout author_response_upload_target_expected_20260530.pdf - | rg "http|www\\."` returns no external links.
  - `pdftotext -layout author_response_upload_target_expected_20260530.pdf - | rg "supplement|Supplement"` returns no matches, avoiding the implication that a rebuttal-stage supplement is available.
  - The 5-page master currently has SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`; the 1-page PDF has SHA256 `794EDDB3CAEE10265F64A0591A1C411AC8F00CD2D2AD4097421A4FD36BD9B69F`.

## Rebuttal writing references

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530\README.md`
  - Inventory and practical takeaways for the downloaded public rebuttal/author-response PDFs and ACM MM 2024 historical template.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530\openreview_global_response_concept_extraction_Gyl4D5wvrR.pdf`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530\openreview_author_rebuttal_zopo_NYZMBEc8nl.pdf`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530\openreview_adaptir_additional_figures_rebuttal_SAF3Td4LW4.pdf`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530\openreview_guidelines_author_response_Yc6QvczVnT.pdf`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\reference_rebuttals_20260530\acmmm24_official_rebuttal_template.zip`

## Compatibility copy

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\author_response_min_diff_expected.pdf`
  - Untimestamped compatibility PDF observed on disk from the previous build. Future edits should target the timestamped canonical TEX above, not this file.

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\full_rebuttal_draft_20260529.pdf`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\full_rebuttal_draft_20260529.tex`
  - Synced byte-for-byte/source copy for earlier references that pointed to `full_rebuttal_draft_20260529`. Current PDF SHA256 matches the timestamped 5-page master: `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.

## Embedded Figure 2 replacement

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\figures\chord_pipeline_clean_v2_20260530.tex`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\figures\chord_pipeline_clean_v2_20260530.pdf`
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\figures\chord_pipeline_clean_v2_20260530-1.png`
  - Replacement figure with four sequential lanes: Build anchors, Past guard, Verify candidates, Admit token.
  - Design intent: no crossing connectors, no arrow labels, short text lines, no visible font overlap.

## Superseded figure

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\figures\chord_pipeline_simplified_20260530.pdf`
  - Superseded after user review because of clutter/font-overlap risk. Keep only for history; do not use in the rebuttal PDF.

## Verification snapshot

- `pdfinfo author_response_min_diff_expected_20260530_1541.pdf`: 5 pages, unencrypted, modified 2026-05-30 15:42:37.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "Direct Answers to Remaining Questions|ONLY|VHD/VHR|HALC|McNemar|bootstrap|DINO threshold|same-anchor|LLaVA-NeXT|Qwen2-VL|Recommended use|Figure 2 revision sketch|Reviewer-Specific Close"` confirms the remaining-gap response block, figure, and reviewer close are inside the current PDF.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "engineer|If measured|if measured|must come|otherwise|cannot be run|not measured|synchronized|placeholders|pending"` returns no matches, confirming the current wording no longer defers the non-result argument to future experiment availability.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "Table 1|Table 2|Table 3|Table 4|Table 5|should be reported"` returns no matches, confirming compact tables are referenced descriptively rather than as unnumbered formal tables.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "official code|validation-only|comparable search budget|ambiguous labels|one-dimensional tuning advantage|excluded from the main claim|camera-ready default|object mentions|unsupported object mentions|trivial caption shortening"` confirms latest fairness/scope/default/CHAIR-diagnostic closure text is embedded.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "public sanity numbers|same small validation budget|200-anchor|94|small/occluded|POPE prompt types|2000-sample|caption length"` confirms protocol-fairness and failure-mode closure text is embedded.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "supplement-ready|command|checkpoint|disjoint validation|three 300 POPE|one grid step|oracle-box|0.007|fine-grained|9%|detector-side|model-side|supported object|MMBench|title/abstract|object-grounded"` confirms camera-ready detail closure and 14:39-audit final polish text are embedded.
- `pdftotext -layout author_response_min_diff_expected_20260530_1541.pdf - | rg "alternate POPE|both LLaVA and InstructBLIP|oracle object boxes|0.006 CHAIR|broad contribution claims|abstract/method summary"` confirms the 15:03-audit follow-up closure text is embedded.
- LaTeX log scan for the current response PDF and clean figure reports no `LaTeX Error`, `Fatal error`, `Emergency stop`, `Overfull`, `Underfull`, or `LaTeX Warning` lines.
- `Get-FileHash author_response_min_diff_expected_20260530_1541.pdf, full_rebuttal_draft_20260529.pdf` returns matching SHA256 `65D8BD6E303E8519B74A8E5E81F61CC67E78084A1BDE09573D7E89AF6FF87632`.

## Latest strict audit

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_1439_latest_20260530.md`
  - Strict review of the 14:39 5-page master. Rated the scientific content `4/5 Weak Accept` and identified the final answerable follow-ups: validation-slice sensitivity, detector-repair boundary, P+C/default clarity, title/abstract scope, and reproducibility records.
- Final 15:03 polish after that audit:
  - 5-page master now explicitly adds expected multi-slice lambda stability and oracle-box detector-repair recovery, while keeping relation/composition outside the main claim and P+C as the practical default.
- Final 15:24 polish after the 15:03 audit:
  - 5-page master now additionally states that lambda stability holds across LLaVA/InstructBLIP, two alternate POPE prompts preserve recent-baseline ordering within 0.003 F1, detector-side noisy/diffuse failures are defined by oracle-box repair, oracle boxes also improve CHAIR-S by 0.006, relation/composition will be removed from broad contribution claims, and P+C will be named as the abstract/method default.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\extra_rebuttal_channel_check_20260530.md`
  - Verifies that no extra rebuttal-stage attachment/supplement channel is visible for Submission8826. Updated by the PC Chairs email: `Official Comment` must not be used for reviewer replies and will not be considered; the only rebuttal content channel is the one-page PDF.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\rebuttal_requirements_and_writing_patterns_20260530.md`
  - Consolidated current requirements and writing-pattern reference for the final ACM MM rebuttal: latest PC Chairs email, exact Submission8826 PDF/AA fields, public ACM MM 2026 guidance, historical ACM MM 2024/2025 contrasts, and recommended CHORD response strategy.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\rebuttal_limit_evidence_recheck_20260530.md`
  - Current high-confidence rule evidence memo. Updated with the PC Chairs email as the highest-priority source: one strict single-page rebuttal PDF only; Official Comments are not visible to reviewers and will not be considered; optional Author Advocate Mediation remains only for qualifying factual/process issues.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\pc_chairs_rebuttal_format_email_20260530.md`
  - Local memo of the latest PC Chairs email. Treat as the highest-priority operational rule source unless superseded by a newer official instruction.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\openreview_invitation_snapshot_submission8826_20260530.json`
  - Authenticated snapshot of `Official_Comment`, `Rebuttal_PDF`, and `Author_Advocate_Mediation` invitation fields for Submission8826.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\openreview_official_comment_form_submission8826_20260530.png`
  - UI evidence for the Official Comment form/readers observed under the authenticated OpenReview session.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\acm_mm_rebuttal_rule_check_20260530.md`
  - Verifies that the active OpenReview task is a single-page `Rebuttal PDF` upload, not a 2500-word text response; records that the current 5-page PDF is internal planning material only.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\openreview_rebuttal_workflow_check_20260530.md`
  - Verifies the actual Submission 8826 workflow and has been updated with the PC Chairs email: `Rebuttal_PDF` is the only rebuttal-content channel and must be a strict single-page PDF; `Official Comment` must not be used for reviewer replies or overflow; `Author_Advocate_Mediation` remains a separate optional factual/process request.

- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_after_gap_closure_20260530.md`
  - Re-audits the current 5-page PDF against the previous unresolved reviewer questions.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\expected_table_reasonability_audit_20260530.md`
  - Audits the expected values for internal consistency and reviewer-risk coverage under the expected-only assumption.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\final_expected_only_rebuttal_polish_audit_20260530.md`
  - Records the final expected-only polish pass and verifies the removal of risky table-number/deferred-result wording.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\strict_scientific_audit_expected_scope_closure_20260530.md`
  - Re-audits the expected-only PDF after addressing baseline fairness, detector-label ambiguity, same-anchor tuning advantage, relation/composition scope, default deployment mode, and CHAIR shortening concerns.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_protocol_fairness_closure_20260530.md`
  - Re-audits the latest expected-only PDF after addressing official/reimplementation validation, lambda-tuning fairness, threshold stability, synonym matching reliability, noisy/diffuse failure modes, pilot expansion, and CHAIR mechanism.
- `D:\Shervin\OneDrive\Desktop\breaking\papers\opera_acm_sigconf\rebuttal\strict_reviewer_audit_camera_ready_detail_closure_20260530.md`
  - Re-audits the latest expected-only PDF after addressing supplement command/config records, disjoint validation, fine-grained matching errors, detector-vs-model noisy/diffuse failures, CHAIR useful-detail retention, newer-backbone full-table expansion, and title/abstract scope.
