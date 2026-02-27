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
  │    E(t, λᵢ) = A_E(λᵢ) · exp(-(t-τg)²/τ_E²) [1] │
  └──────────────────────────────────────────────────┘
        │                                  │
  3-D plot I=E²(t,λ) — initial     3-D plot I=E²(t,λ) — initial
  3-D plot I=E²(t,λ) — propagated  3-D plot I=E²(t,λ) — propagated
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

### Pulse propagation — stored as **field amplitude**, plotted as intensity

`build_pulse()` propagates and stores **field amplitude** arrays `E_init`, `E_prop`:

```
E(t, λᵢ) = A_E(λᵢ) · exp(−(t − τg(λᵢ))² / τ_E²)      [1]
```

| Symbol | Meaning |
|--------|---------|
| `A_E(λᵢ)` | Field amplitude spectral weight — Gaussian with FWHM_field = `Δλ_I · √2` |
| `τ_E` | Field Gaussian parameter, derived from the input intensity FWHM via TBP |
| `exp(−t²/τ_E²)` | Temporal **field** Gaussian — squaring gives the intensity envelope |
| `τg(λᵢ)` | Relative group delay of spectral component λᵢ (reference = λ₀) |

The intensity at (λᵢ, t) is recovered by squaring: `I(t,λᵢ) = E(t,λᵢ)²`.

The 3D pulse plots labeled **"Intensity I"** display `E_prop²` — squaring
is done inside `plot_pulse_3d`. This is what a photodetector measures.

Key relationships between field and intensity Gaussians:

```
field:     E(t) ∝ exp(−t²/τ_E²)          FWHM_field = τ_E · 2√(ln 2)   ≈ 1.18 τ_E
intensity: I(t) = E² ∝ exp(−2t²/τ_E²)   FWHM_I     = τ_E · √(2 ln 2)  ≈ 0.83 τ_E

→  FWHM_I = FWHM_field / √2
```

The input `Δλ_I` is the **intensity** spectral FWHM (as measured on a spectrometer).
`spectral_nm_to_tau` converts it to the field τ_E via the TBP:
`Δt_I · Δν_I = 2ln2/π ≈ 0.441`  (intensity FWHM × intensity FWHM, TL pulse).

The spectral weight `A_E` has field FWHM = `Δλ_I · √2` so that `A_E² = A_I`
recovers the measured intensity spectral envelope with FWHM = `Δλ_I`.
The spectral grid is ±3.5 σ_E around λ₀ (field amplitude < 0.3 % of peak at edges).

### CARS convolutions — also at **field amplitude** level

The Raman coherence driven in a CARS process is:

```
ρ(Ω, t) ∝ E_pump(t) × E_Stokes*(t)      (field × field)
```

`compute_conv1` uses `E_prop` directly — no square root needed — to form the
physically correct product:

```
C₁(Ω, t) = Σₖ E_pump(νₖ, t) · E_Stokes(νₖ − Ω, t)      [2]
```

Using intensity arrays (I × I) instead would square the coherence and artificially
narrow its spectral width by √2, overestimating the resolution.

Similarly, the probe step in `compute_conv2_2d` uses the pump field amplitude:

```
C₂(ν_AS, t) = ∫ C₁(Ω, t) · E_pump(ν_AS − Ω, t) dΩ       [3]
```

where `ν_AS = ν_pump_center + Ω` (~15 292 cm⁻¹ ≈ 654 nm).

The 3D plots labeled **"Excitation amplitude"** and **"AS amplitude"**
show these field-amplitude quantities. To convert to observable intensities
(e.g. photon counts), square the values.

### Summary

| Array / Plot | Quantity | Gaussian exponent |
|---|---|---|
| `E_init`, `E_prop` | Field amplitude E(t,λ) | exp(−t²/τ_E²) |
| Pulse 3D plots z-axis | Intensity I = E² | exp(−2t²/τ_E²) |
| `C₁(Ω, t)` | Raman coherence amplitude ∝ E_pump·E_Stokes | — |
| C₁ 3D plots z-axis | Excitation amplitude | — |
| `C₂(ν_AS, t)` | Anti-Stokes field amplitude ∝ E_pump·C₁ | — |
| C₂ 3D plots z-axis | AS amplitude | — |

---

## Normalization policy

The only normalization in the pipeline is `A /= A.max()` inside `build_pulse()`,
applied once to the **initial spectral field envelope** before any propagation.
All downstream quantities — `E_prop`, `C₁`, `C₂` — carry physically correct absolute
amplitudes thereafter.

Consequence: more chirping (more glass) means lower peak amplitude and lower peak signal.
This is the intended behaviour: the simulation shows the effect of chirping on **power
distribution**, not just on spectral shape.

---

## Delay scan — what "optimal delay" means here

Each scenario is computed at two Stokes delays:

### Zero delay (`delay_ps = 0`)
The Stokes pulse is not time-shifted relative to the pump. Because both pulses are centred
on their respective group-delay grids (reference = group delay of the central wavelength),
zero delay means their temporal centres coincide. This is the natural starting point.

### Optimal delay (`find_optimal_delay`)
`find_optimal_delay` scans a range of Stokes time offsets δt and selects the one that
**maximises the total integrated C₁ signal** — i.e., `Σ_{Ω,t} C₁(Ω, t)`.

This is a **signal-strength criterion**, not a resolution criterion. It answers:
> At what relative timing do the two chirped pulses overlap the most in time,
> producing the strongest Raman excitation?

For perfectly matched chirp rates (same glass, same wavelength) the answer is δt ≈ 0.
When chirp rates are mismatched — as in Scenario B where the pump has only 5 cm of glass
while the Stokes has 15 cm — the centre of mass of the pump's group-delay sweep arrives
at a slightly different time than the Stokes's, and the optimal delay shifts accordingly.

**What this delay does NOT represent:**
It is not the delay that minimises the spectral width of C₁(Ω) (best resolution).
For mismatched chirp rates those two optima differ.  The comparison plot
(zero delay vs. optimal delay) therefore shows **the effect of temporal recentering on
signal strength and spectral shape**, not a resolution-optimised result.
The RMS values printed alongside are the resolution at each delay, for reference.
