# Changelog

## 2026-02-26 — Initial rewrite of resolution.py

### Added
- Standalone S-TIH6 Sellmeier dispersion (`n_stih6`, `group_index`, `group_delay`) — no dependency on `pulse.py`
- Component-wise propagation: each spectral slice shifted by its group delay → `build_pulse()`
- 2-D time–wavelength–intensity maps before and after glass propagation
- First convolution C₁(Ω, t): pump ⊗ Stokes cross-correlation on shared wavenumber/time grid
- Optimal-delay scan: minimises RMS of integrated Raman excitation spectrum
- Second convolution C₂: degenerate probe broadening via convolution with pump spectral marginal
- Spectral projections with centroid and RMS width (cm⁻¹)
- Centred comparison plot (zero-delay vs optimal-delay)
- All 3-D plots saved as PDF + interactive HTML (plotly)

### Removed
- Old `Resolution` class and its `run()` pipeline
- Dependency on `Pulse` class from `pulse.py`

## 2026-02-26 — Updated parameters, spectral-nm inputs, two scenarios

### Changed
- Beam labels corrected to standard CARS convention:
  pump = 803.31 nm (higher ν), Stokes = 1041.22 nm (lower ν)
- Inputs now spectral intensity FWHM [nm], not temporal FWHM [fs];
  added `spectral_nm_to_tau()` and `temporal_fwhm_fs()` helpers
- Convolution sign fixed: `shifted_wn = pump_wn − Ω` (standard CARS)
- `output_dir` now a required argument to all plot/save functions (no module-level default)
- Extracted `run_scenario(pump_mm, stokes_mm, output_dir, label)` from `main()`

### Added
- Two-scenario runner in `SCENARIOS` list:
  - Scenario A: pump 150 mm · Stokes 150 mm → `plots/resolution/`
  - Scenario B: pump  50 mm · Stokes 150 mm → `plots/resolution_pump5cm/`

### Results (new parameters)
- Raman centre: 2845 cm⁻¹ (ν_pump − ν_Stokes ✓)
- Scenario A optimal C₁ RMS: 19.3 cm⁻¹ → after probe: 79.7 cm⁻¹
- Scenario B optimal C₁ RMS: 41.6 cm⁻¹ → after probe: 87.7 cm⁻¹
- Resolution dominated by pump spectral width (11.77 nm ≈ 182 cm⁻¹)
