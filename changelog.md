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
