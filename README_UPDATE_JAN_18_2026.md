# README Update - January 18, 2026

## Changes Made

Updated README.md to reflect Zenodo publication instead of arXiv submission.

### 1. Badge Updated (Line 5)
**OLD:**
```markdown
[![arXiv](https://img.shields.io/badge/arXiv-Submitted-b31b1b.svg)](https://arxiv.org/)
```

**NEW:**
```markdown
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18292199-blue)](https://doi.org/10.5281/zenodo.18292199)
```

### 2. Project Structure Comment (Line 171)
**OLD:**
```markdown
└── papers/                   # arXiv paper (private)
```

**NEW:**
```markdown
└── papers/                   # Published paper (Zenodo)
```

### 3. Documentation Section (Line 178)
**OLD:**
```markdown
- **arXiv Paper** - "QPL: A Relations-First Programming Language for Measurement-Based Quantum Computing" (submitted January 2026)
```

**NEW:**
```markdown
- **[Published Paper](https://doi.org/10.5281/zenodo.18292199)** - "QPL: A Relations-First Programming Language for Measurement-Based Quantum Computing" (Zenodo preprint, January 2026)
```

---

## Why These Changes

1. **Zenodo is published** - arXiv rejected, Zenodo accepted and published with DOI
2. **Professional credibility** - DOI badge looks better than "Submitted"
3. **Clickable links** - Users can now click directly to the paper
4. **Accurate status** - No longer claiming arXiv submission

---

## Verification

Run to check updates:
```bash
grep -n "Zenodo\|DOI" README.md
```

Should show:
- Line 5: Zenodo badge with DOI
- Line 171: Published paper comment
- Line 178: Published Paper link with DOI

---

## Next Steps

**Optional but recommended:**
- [ ] Commit and push to GitHub
- [ ] GitHub README will automatically show Zenodo badge
- [ ] Anyone visiting repo sees published paper DOI

**Commit message suggestion:**
```
Update README with Zenodo publication (DOI: 10.5281/zenodo.18292199)

- Replace arXiv badge with Zenodo DOI badge
- Update documentation links to published paper
- Reflect complete Stage 0-3 implementation status
```

---

**Status:** README updated ✅
**DOI:** https://doi.org/10.5281/zenodo.18292199
**Date:** January 18, 2026, 18:50 GMT
