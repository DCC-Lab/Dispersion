# Dispersion — spectral-focusing CARS simulation

Simulation and analysis of laser pulse chirping through optical glass for
**spectral-focusing CARS** (Coherent Anti-Stokes Raman Scattering) microscopy.
Target application: imaging myelin CH₂ vibrations (~3200 cm⁻¹) with a
803 nm pump + 1041 nm Stokes two-colour setup.

## Files

| File | Purpose |
|------|---------|
| `resolution.py` | Main simulation: propagation, convolutions, resolution readout |
| `pulse.py` | Core `Pulse` class (1-D FFT-based propagation) — reference implementation |
| `3d_shape.py` | Analytical Gaussian visualisation (`ShapePlotter`) — decoupled from physics |
| `testDispersion.py` | Unit tests for FFT calibration and time-bandwidth product |
| `changelog.md` | Change history |
| `agents.md` | Design notes for AI session continuity |
| `TODO.md` | Pending tasks |

## Quick start

```bash
# Install dependencies (if needed)
pip install numpy scipy matplotlib plotly

# Run the full simulation
python resolution.py
```

Outputs are saved to `plots/resolution/` and `plots/resolution_pump5cm/` as `.pdf` and `.html` files.

## Physics pipeline (resolution.py)

```
Pump (Δλ_I, λ₀, glass)          Stokes (Δλ_I, λ₀, glass)
        │                                  │
        ▼                                  ▼
  build_pulse()                      build_pulse()
  ┌──────────────────────────────────────────────────┐
  │  For each λᵢ in spectral grid:                   │
  │    group_delay(λᵢ) = d · n_g(λᵢ) / c            │
  │    I(t, λᵢ) = A(λᵢ) · exp(-2(t-τg)²/τ_E²)  [1] │
  └──────────────────────────────────────────────────┘
        │                                  │
  3-D plot I(t,λ) — initial        3-D plot I(t,λ) — initial
  3-D plot I(t,λ) — propagated     3-D plot I(t,λ) — propagated
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              compute_conv1(delay_ps=0)    [2]
              compute_conv1(delay_ps=opt)  ← find_optimal_delay()
              3-D plot C₁(Ω, t) — excitation amplitude
              comparison plot C₁ (zero vs optimal)
                       │
                       ▼
              compute_conv2_2d()   [3]
              3-D plot C₂(ν_AS, t) — anti-Stokes amplitude
              spectral_stats()  → centroid, RMS width
              plot_projection()
              comparison plot C₂ (zero vs optimal)
```

## Key parameters (top of resolution.py)

```python
# Pump: shorter wavelength = higher frequency (standard CARS convention)
PUMP_FWHM_NM    = 11.77     # Spectral intensity FWHM [nm]
PUMP_LAMBDA0_NM = 803.31    # Central wavelength [nm]

# Stokes: longer wavelength = lower frequency
STOKES_FWHM_NM    = 8.92
STOKES_LAMBDA0_NM = 1041.22

GLASS_MATERIAL = 'stih6'    # S-TIH6 Schott glass

# Two scenarios run automatically:
SCENARIOS = [
    (150, 150, 'plots/resolution',         'Scenario A — both 15 cm'),
    ( 50, 150, 'plots/resolution_pump5cm', 'Scenario B — pump 5 cm'),
]
```

---

## Physical conventions — field vs. intensity

This section documents a distinction that is critical for interpreting the
simulation output correctly.

### Pulse propagation — stored as **intensity**

`build_pulse()` stores `I_init` and `I_prop` as **optical intensity** arrays:

```
I(t, λᵢ) = A(λᵢ) · exp(−2(t − τg(λᵢ))² / τ_E²)      [1]
```

| Symbol | Meaning |
|--------|---------|
| `A(λᵢ)` | Intensity spectral weight — Gaussian with FWHM = input `Δλ_I` |
| `τ_E` | Field Gaussian parameter, derived from input via TBP = Δt_I·Δν_I = 2ln2/π |
| `exp(−2t²/τ_E²)` | Temporal **intensity** Gaussian (factor −2 ↔ field squaring) |
| `τg(λᵢ)` | Relative group delay of spectral component λᵢ (reference = λ₀) |

The field amplitude at (λᵢ, t) is therefore `E(t,λᵢ) = √I(t,λᵢ)`.

The 3D plots labeled **"Intensity I (norm.)"** show exactly this intensity
representation. This is what a photodetector measures and the natural quantity
for visualisation.

Note that τ_E is the **field** Gaussian parameter, not the intensity one:

```
field:     E(t) ∝ exp(−t²/τ_E²)         FWHM_field = τ_E · 2√(ln 2)   ≈ 1.18 τ_E
intensity: I(t) ∝ exp(−2t²/τ_E²)        FWHM_I     = τ_E · √(2 ln 2)  ≈ 0.83 τ_E

→ FWHM_I = FWHM_field / √2
```

The input `Δλ_I` is the **intensity** spectral FWHM (as measured on a spectrometer).
Its corresponding temporal FWHM is `Δt_I = 2ln2 / (π·Δν_I)` (transform-limited).
Both are intensity-level quantities, consistent with the TBP ≈ 0.441.

### CARS convolutions — operating at **field amplitude** level

The Raman coherence driven in a CARS process is:

```
ρ(Ω, t) ∝ E_pump(t) × E_Stokes*(t)      (field × field)
```

`compute_conv1` therefore operates on **field amplitudes** √I, not on intensities:

```
C₁(Ω, t) = Σₖ √I_pump(νₖ, t) · √I_Stokes(νₖ − Ω, t)      [2]
```

Using I_pump × I_Stokes (intensity product) instead would square the coherence
and artificially narrow its spectral width by √2, overestimating the resolution.

Similarly, the probe step in `compute_conv2_2d` uses the pump field amplitude:

```
C₂(ν_AS, t) = ∫ C₁(Ω, t) · √I_pump(ν_AS − Ω, t) dΩ       [3]
```

where `ν_AS = ν_pump_center + Ω` (~15 292 cm⁻¹ ≈ 654 nm).

The 3D plots labeled **"Excitation ampl. (norm.)"** and **"AS ampl. (norm.)"**
reflect these field-amplitude quantities. To convert to observable intensities
(e.g. photon counts), one would square these values — but for spectral resolution
analysis (RMS width of C₁, C₂) the amplitude form suffices and is standard.

### Summary

| Array / Plot | Quantity | Factor |
|---|---|---|
| `I_init`, `I_prop` | Optical intensity I(t,λ) | ∝ \|E\|² |
| Pulse 3D plots z-axis | Intensity (norm.) | ∝ \|E\|² |
| `C₁(Ω, t)` | Raman excitation amplitude | ∝ E_pump·E_Stokes |
| C₁ 3D plots z-axis | Excitation ampl. (norm.) | ∝ E_pump·E_Stokes |
| `C₂(ν_AS, t)` | Anti-Stokes field amplitude | ∝ E_pump·C₁ |
| C₂ 3D plots z-axis | AS ampl. (norm.) | ∝ E_pump·C₁ |
