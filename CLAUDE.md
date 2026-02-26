# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Simulation and analysis of laser pulse dispersion through optical components, in the context of CARS (Coherent Anti-Stokes Raman Scattering) spectroscopy using 805 nm + 1045 nm two-color pulses.

## Commands

Run all tests:
```bash
python -m unittest testDispersion.py
```

Run a single test:
```bash
python -m unittest testDispersion.TestFFTAndPulses.test08timeBandwithProductOfPulseAndFFT
```

Run a script directly:
```bash
python pulse.py
python 3d_shape.py
```

## Architecture

### Core simulation: `pulse.py`

`Pulse(𝛕, 𝜆ₒ)` is the main 1D pulse class. It stores the electric **field** (not intensity) as a Gaussian envelope with carrier: `exp(-(t²/𝛕²)) * cos(𝝎ₒ*t)`. The `𝛕` parameter is the Gaussian parameter of the **electric field**.

Propagation (`propagate(d, indexFct)`) works in frequency space: FFT → apply phase `exp(i * 2π/λ * n(λ) * d)` → iFFT. Materials are Sellmeier-equation methods on the class (`silica`, `bk7`, `stih6`, `sf10`, `water`).

Key numeric convention: `spectralWidth` is RMS in frequency (Hz), `temporalWidth` is RMS in time (s), and `timeBandwidthProduct = 2π * Δf_rms * Δt_rms` (transform limit ≈ 0.44 for FWHM-based Gaussian, but the class uses RMS).

### 3D visualization: `3d_shape.py`

`ShapePlotter(sigma_t, sigma_lambda, t0, lambda0, unit)` visualizes pulses as 2D Gaussians in time–wavelength (or wavenumber) space. It is decoupled from `Pulse` — it is purely for visualization, not simulation. The `unit` parameter accepts `'nm'` or `'cm-1'`.

### Resolution analysis: `resolution.py`

Standalone script (not a class) computing CARS spectral resolution trade-offs. Key physical context: 1045 nm fixed pump, tunable 805 nm Stokes, targeting CH₂ vibrations at ~3200 cm⁻¹ (myelin). Explores stretching a 100 fs pulse to ~6 ps using glass dispersion.

### Tests: `testDispersion.py`

Nine `unittest` cases validating FFT calibration, RMS width calculation, and time-bandwidth product (both transform-limited and with chirp via a TL factor). No mocking — all tests are pure numpy.

## Key Conventions

- **Unicode variable names** are used throughout for physics (𝛕, 𝜆ₒ, 𝝎ₒ, π, etc.).
- **Gaussian parameter is for the field**, not the intensity. Intensity FWHM = `𝛕 * √2 * 2√(ln2)`.
- All internal units are SI (m, s, Hz) except plot labels which display ps, fs, nm, THz.
- `dispersion.py` is a **deprecated** earlier version of `pulse.py` — do not extend it.
