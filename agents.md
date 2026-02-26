# Agents — session notes for AI continuity

## Branch: 3d-propagation

## Current state (2026-02-26)
`resolution.py` has been fully rewritten.  The key design decisions are recorded here
so that any future agent can pick up without re-reading all code.

### Architecture decisions
| Decision | Rationale |
|----------|-----------|
| No `Pulse` class import | User requested own propagation, independent of `pulse.py` |
| Component-wise group delay | First-order dispersion picture; gives intuitive chirp tilt in time–wavelength space |
| Vectorised interp in conv1 | `interp1d(..., axis=0)` over spectral dimension avoids inner Python loop over wavenumber |
| `fftconvolve` for conv2 | Fast; the pump kernel is narrow and centred, so `mode='same'` is correct |
| PDF + plotly HTML | PDF for publication, HTML for interactive 3-D rotation |
| `TkAgg` backend | Interactive windows; change to `'Agg'` for headless / batch runs |

### Physical conventions
- `τ = FWHM_I / sqrt(2·ln2)` — field Gaussian parameter from intensity FWHM
- Spectral FWHM: `Δλ = λ₀²/c · 2·ln2/(π·FWHM_I)`
- Wavenumber: `ν [cm⁻¹] = 1e7 / λ [nm]`
- Raman axis: 2500–3800 cm⁻¹ (CH₂ vibrations, myelin)

### Known limitations / future work
- Propagation is first-order (group delay only); GVD-induced temporal broadening of each
  narrow-band slice is not yet included.
- Optimal-delay scan is a brute-force 1-D grid search (150 points, ±3 ps); could be
  replaced by golden-section search for speed.
- Only S-TIH6 glass is implemented; `n_glass()` dispatcher is ready for extension.
