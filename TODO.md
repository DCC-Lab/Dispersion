# TODO

## High priority
- [ ] Add GVD-induced temporal broadening within each narrow-band spectral slice
      (currently only group delay is applied; second-order dispersion is ignored)
- [ ] Verify conv1 output physically: for matched chirp rates the ridge should be
      horizontal in the (t, Ω) plane
- [ ] Add wavenumber axis option for pulse 3-D plots (x-axis in cm⁻¹ instead of nm)

## Medium priority
- [ ] Replace brute-force delay scan with golden-section search (faster)
- [ ] Add BK7 and silica to `n_glass()` dispatcher (coefficients already in pulse.py)
- [ ] CLI argument parsing so parameters can be overridden without editing the file
- [ ] Add intermediate propagation snapshots as an optional mode (every N mm of glass)

## Low priority / nice to have
- [ ] Export plotly figures with camera presets (top-down, side, 45°)
- [ ] Add a 2-D heatmap version of each 3-D plot (faster to generate for batch runs)
- [ ] Unit tests for `n_stih6`, `group_index`, `build_pulse`, `spectral_stats`
- [ ] Notebook version with ipywidgets sliders for real-time parameter tuning
