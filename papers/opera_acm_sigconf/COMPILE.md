# Compile Guide

This paper is now aligned to the ACM official conference template workflow:

- Main source: `sample-sigconf.tex`
- Bibliography: `sample-base.bib`
- Main output: `sample-sigconf.pdf`
- Supplementary source: `supplementary.tex`

## Main Paper

Compile from `papers/opera_acm_sigconf`:

```powershell
latexmk -g -xelatex -interaction=nonstopmode -file-line-error sample-sigconf.tex
```

The generated PDF is:

```text
sample-sigconf.pdf
```

## Clean Main Build Artifacts

```powershell
latexmk -C sample-sigconf.tex
```

If you want to rebuild from scratch right away:

```powershell
latexmk -C sample-sigconf.tex
latexmk -g -xelatex -interaction=nonstopmode -file-line-error sample-sigconf.tex
```

## Supplementary Material

Compile separately:

```powershell
latexmk -g -xelatex -interaction=nonstopmode -file-line-error supplementary.tex
```

Output:

```text
supplementary.pdf
```

## Notes

- The paper body is written directly inside `sample-sigconf.tex`.
- Do not use `main.tex` or `references.bib`; they were part of the old layout.
- If ACM rights-form metadata becomes available later, update the copyright and conference fields near the top of `sample-sigconf.tex`.
