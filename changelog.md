# Changelog

## 2026-03-02 — Increase spectral grid resolution (N_SPEC 80 → 400)

### Changed
- **`N_SPEC = 400`** (was 80): spectral grid points per beam in `resolution.py`.
  Root cause of the discrete "rib" artefacts in `pump_step0_initial` and
  `pump_step1_propagated` was that adjacent slices were separated by ~53 fs in
  group delay, exceeding the individual field width (~48 fs) so each slice was
  visually distinct.  With 400 points ΔGD ≈ 10 fs ≪ τ_field → smooth continuous
  Gaussian surfaces in both PDF (pcolormesh) and interactive HTML (plotly surface).
  Change propagates automatically into `compute_conv1`, `compute_conv2_2d`, and
  `run_delay_scan` (no other code changes required).
- Estimated runtime impact: `run_delay_scan` ~80 s (was ~16 s); full pipeline ~200 s.

## 2026-03-02 — Delay-scan power & spectral-centre curves (Scenario A)

### Added
- **`run_delay_scan(pump, stokes, raman_axis)`**: scans Stokes delay τ over
  `[−DELAY_SCAN_PS_RANGE, +DELAY_SCAN_PS_RANGE]` in `DELAY_SCAN_N` steps.  At each τ
  records total C₁ and C₂ field amplitude (∫∫ C dΩ dt, consistent with
  `find_optimal_delay`) plus the power-weighted spectral centre for each layer.
  Prints a summary table (peak delay, centroid, RMS width, accessible spectral range).
  Intended for Scenario A only.
- **`plot_delay_scan(delays, power, center, …)`**: two-row figure (PDF + plotly HTML).
  Row 1 — normalised signal power vs. Stokes delay with fill-under shading, vertical
  markers for peak & centroid, annotated RMS width.
  Row 2 — power-weighted spectral centre vs. delay for the thresholded region
  (power > 1 % of peak), coloured scatter (plasma) by normalised power, accessible
  range bounds, and tunable-span annotation.  Uses `constrained_layout` to eliminate
  the tight-layout / colorbar conflict.
- **Steps 11–12 in `run_scenario`** (Scenario A only, `pump_glass_mm == 150`):
  calls `run_delay_scan` then `plot_delay_scan` for both C₁ and C₂.
  Output files: `step11_c1_delay_scan.{pdf,html}` and `step12_c2_delay_scan.{pdf,html}`
  in `plots/resolution/`.

### Results (Scenario A, both 15 cm S-TIH6, 150-point delay scan ±8 ps)
- **C₁ (Raman excitation)**:
  - Power peak: τ = −0.054 ps (≈ 0, as expected for symmetric glass lengths)
  - RMS width of power curve: **0.631 ps** → accessible delay range
  - Accessible Raman range: **2526 – 3183 cm⁻¹** (tunable span = 657 cm⁻¹)
  - Power-weighted spectral RMS: 114.8 cm⁻¹
- **C₂ (anti-Stokes signal)**:
  - Power peak: τ = −0.054 ps (same as C₁)
  - RMS width: **0.455 ps** (narrower — second pump interaction restricts window)
  - Accessible AS range: **14971 – 15771 cm⁻¹** (tunable span = 799 cm⁻¹)
  - Power-weighted spectral RMS: 148.8 cm⁻¹

### Physics interpretation
- The spectral centre shifts **linearly** with delay (spectral focusing tuning rate
  ≈ 330 cm⁻¹/ps for C₁, ≈ 400 cm⁻¹/ps for C₂).
- C₂ has a narrower power window but a wider absolute AS span because the probe pump
  dispersion adds to the spectral shift on the anti-Stokes axis.

## 2026-02-27 — Fix pulse plot aliasing; document marginal intensity physics

### Changed
- **`plot_pulse_3d`**: replaced the matplotlib 3-D surface (`plot_surface` with
  `rcount=60, ccount=60`) with a 2-D heatmap (`pcolormesh`) for the PDF output.
  The old surface subsampled the 1000-point time axis to 60 columns, causing the
  narrow (~15 sample) TL peak to alias into separate phantom peaks in the PDF.
  The interactive HTML output (plotly) is unchanged — it renders all data points
  and always showed the correct continuous Gaussian.
- **`marginal_temporal_intensity` docstring**: added a full physics derivation
  explaining why the marginal peak drops even though each individual spectral
  slice retains its amplitude after glass propagation, with the "musicians playing
  in unison vs. sequentially" analogy and the formula 1/n_eff.
- **`readme.md`** — "Glass propagation and chirp physics" section: added
  "Physical intuition" paragraph with the same analogy, a comparison table, and
  the formula; also noted the measured n_eff ≈ 12 for the pump in Scenario A.

## 2026-02-27 — Cleanup: fix stale docstrings; update agents.md

### Changed
- **`compute_conv1` docstring**: removed stale `E = √I` and `I_prop stores intensity`
  phrasing. Arrays have been named `E_prop` and store field amplitudes directly since the
  field-amplitude refactor; no square root is ever taken.
- **`compute_conv2_2d` docstring**: removed stale `E_pump = √I_pump` phrasing; same reason.
- **`_interp_rows`**: added docstring and renamed parameter `I_src → E_src` for consistency.
- **`agents.md`**: updated to 2026-02-27; added normalization-policy and delay-criterion
  decisions to the architecture table.
- **README.md**: stale `(norm.)` labels removed from field/intensity summary table; added
  Normalization policy section and Delay scan section explaining signal-strength criterion.

## 2026-02-27 — Optimal-delay criterion reverted to max-signal; README documented

### Changed
- `find_optimal_delay` reverted to maximising total integrated C₁ signal (temporal-overlap
  criterion). The docstring now explicitly states this is a **signal-strength criterion**,
  not a resolution criterion, and notes the distinction for mismatched chirp rates.
- README: added **Normalization policy** section and **Delay scan** section explaining what
  zero delay and optimal delay represent and what the optimal delay does NOT mean.

## 2026-02-27 — Remove post-initial normalizations; show chirping power reduction

### Changed
- **Physics fix (`compute_conv2_2d`)**: removed time-varying pump kernel normalization
  (`P_norm = P_kern / P_kern.sum()`). The kernel now carries actual pump field amplitudes,
  so chirping-induced peak amplitude reduction correctly propagates into C₂.
- **All display plots**: removed per-array normalization (`/ array.max()`) from
  `plot_pulse_3d`, `plot_conv1_3d`, `plot_conv2_3d`, `plot_projection`, and `plot_comparison`.
  Plots now show absolute amplitudes — chirped pulses and their convolutions appear smaller
  than transform-limited ones, which is the physically correct behaviour.
- Updated axis labels: `'Intensity I (norm.)'` → `'Intensity I'`, `'Excitation ampl. (norm.)'`
  → `'Excitation amplitude'`, `'AS ampl. (norm.)'` → `'AS amplitude'`,
  `'Intensity (norm.)'` → `'Amplitude'`.
- **Normalization policy**: the only normalization in the pipeline is `A /= A.max()` in
  `build_pulse()`, applied once to the initial spectral field envelope. All downstream
  quantities (E_prop, C₁, C₂) reflect physical amplitudes thereafter.

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

## 2026-02-26 — Propagation also on field amplitude (E_prop replaces I_prop)

### Changed
- `build_pulse`: spectral weight A is now the **field amplitude** envelope:
  - `sigma_E = FWHM_I / (2√ln2)` (was `FWHM_I / (2√(2ln2))` for intensity σ)
  - `A_E = exp(-0.5*(λ-λ0)²/σ_E²)` → FWHM_field = FWHM_I·√2; A_E² gives FWHM_I ✓
  - Temporal Gaussian changed from `exp(-2t²/τ_E²)` (intensity) to `exp(-t²/τ_E²)` (field)
  - Dict keys renamed: `I_init` → `E_init`, `I_prop` → `E_prop`
- Spectral grid now ±3.5 σ_E (wider by √2; field amplitude < 0.3 % at edges)
  → GD spread reported wider by √2 (more spectral components included)
- `plot_pulse_3d`: squares E_2d before display (`I = E²`); z-axis label unchanged
- `compute_conv1`: `E_prop` used directly (no `np.sqrt` needed any more)
- `compute_conv2_2d`: same — pump kernel from `E_prop` directly
- README "Physical conventions" section updated throughout

### Results (unchanged physics, as expected)
- C₁ RMS: 27.1 cm⁻¹, C₂ RMS: 42.7 cm⁻¹ (within numerical precision of previous run)
