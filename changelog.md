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

## 2026-02-26 — Time-resolved C₂, anti-Stokes axis, comparison plots, TBP

### Added
- `time_bandwidth_product()`: TBP = Δt_FWHM[fs] × Δν_FWHM[THz] for TL Gaussian (≈ 0.441);
  printed in console and included in initial-pulse 3D plot titles
- `compute_conv2_2d()`: time-resolved probe convolution; output on anti-Stokes absolute
  wavenumber axis ν_AS = ν_pump_center + Ω (~15 292 cm⁻¹ ≈ 654 nm)
- `plot_conv2_3d()`: 3D surface of C₂(ν_AS, t) saved as PDF + interactive HTML;
  dual cm⁻¹ / nm tick labels on wavenumber axis; plotly hover shows both units
- `comparison_c1.pdf`: centred C₁ marginals (zero-delay vs optimal-delay) on Raman axis
- `comparison_c2.pdf`: centred C₂ marginals on anti-Stokes axis

### Changed
- `compute_conv2` replaced by `compute_conv2_2d` (time-resolved, anti-Stokes axis)
- `plot_comparison` generalised: accepts `axis`, `xlabel`, `step_name`, `suptitle` params
- `plot_projection` accepts optional `xlabel` kwarg (default `'Raman shift (cm⁻¹)'`)
- `run_scenario` updated to 9 steps

### Results (session 3)
- Pump TBP = 0.441, Stokes TBP = 0.441 (TL Gaussians ✓)
- Scenario A: ν_AS = 15292 cm⁻¹ (654 nm), C₂ RMS = 30 cm⁻¹
- Scenario B: ν_AS = 15292 cm⁻¹ (654 nm), C₂ RMS = 92 cm⁻¹ (less pump chirp)

## 2026-02-26 — Fix convolutions to field amplitude; document field vs. intensity

### Fixed
- `compute_conv1`: was computing I_pump × I_Stokes (intensity product), which squares the
  Raman coherence and underestimates excitation bandwidth by √2. Now computes
  √I_pump × √I_Stokes (field amplitude product) — the physically correct driving term
  ρ(Ω,t) ∝ E_pump × E_Stokes.
- `compute_conv2_2d`: pump probe kernel changed from I_pump to √I_pump (field level),
  consistent with the C₂ = E_pump × coherence amplitude interpretation.
- Z-axis labels updated: "Intensity I (norm.)" for pulse plots, "Excitation ampl. (norm.)"
  for C₁, "AS ampl. (norm.)" for C₂.

### Added / changed
- README.md: new "Physical conventions — field vs. intensity" section documenting exactly
  what each stored array and 3D plot axis represents, with the key formulas.

### Results after fix (Scenario A, both 15 cm)
- C₁ RMS (optimal delay): 26.9 cm⁻¹ (was 19.3 cm⁻¹ with intensity product — ratio √2 ✓)
- C₂ RMS: 42.5 cm⁻¹ (was 30.2 cm⁻¹ — ratio √2 ✓)
