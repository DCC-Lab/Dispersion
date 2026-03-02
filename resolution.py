"""
resolution.py — Spectral-focusing CARS resolution simulation

Step-by-step chirping of pump (803 nm) and Stokes (1041 nm) pulses through
S-TIH6 glass, followed by the pump⊗Stokes cross-correlation (Raman excitation)
and degenerate probe broadening, with spectral resolution readout.

Inputs: spectral FWHM of the intensity Gaussian [nm] + central wavelength [nm].
Runs two glass-length scenarios automatically and saves each to its own folder.

All 3-D plots are saved as PDF (matplotlib) + interactive HTML (plotly).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # interactive backend; change to 'Agg' for headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False
    print("WARNING: plotly not installed — HTML output disabled. Run: pip install plotly")

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit these to change inputs)
# ─────────────────────────────────────────────
# Pump: shorter wavelength = higher frequency (standard CARS convention)
PUMP_FWHM_NM    = 11.77     # Pump spectral intensity FWHM [nm]
PUMP_LAMBDA0_NM = 803.31    # Pump central wavelength [nm]

# Stokes: longer wavelength = lower frequency
STOKES_FWHM_NM    = 8.92      # Stokes spectral intensity FWHM [nm]
STOKES_LAMBDA0_NM = 1041.22   # Stokes central wavelength [nm]

GLASS_MATERIAL = 'stih6'    # Currently only stih6 supported

N_SPEC   = 80     # Spectral grid points per beam
N_TIME   = 1000   # Time grid points

RAMAN_MIN_CM1 = 2500   # Raman axis lower bound [cm⁻¹]
RAMAN_MAX_CM1 = 3800   # Raman axis upper bound [cm⁻¹]
N_RAMAN       = 300    # Points on Raman axis

DELAY_SCAN_PS_RANGE = 8.0   # ± range for optimal-delay scan [ps]
DELAY_SCAN_N        = 150   # Number of delay points to scan

# Scenarios — (pump_glass_mm, stokes_glass_mm, output_dir)
SCENARIOS = [
    (150, 150, 'plots/resolution',         'Scenario A — pump 15 cm · Stokes 15 cm'),
    ( 50, 150, 'plots/resolution_pump5cm', 'Scenario B — pump  5 cm · Stokes 15 cm'),
]

# ─────────────────────────────────────────────
#  PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
c = 2.99792458e8   # speed of light [m/s]


# ─────────────────────────────────────────────
#  GLASS DISPERSION  (S-TIH6 Sellmeier)
# ─────────────────────────────────────────────

def n_stih6(wl_m):
    """Refractive index of S-TIH6.  wl_m: wavelength in metres (scalar or array)."""
    x = np.asarray(wl_m, dtype=float) * 1e6          # → µm
    x = np.clip(x, 0.3, 2.5)
    n2 = (1
          + 1.77227611 / (1 - 0.0131182633 / x**2)
          + 0.34569125 / (1 - 0.0614479619 / x**2)
          + 2.40788501 / (1 - 200.753254   / x**2))
    return np.sqrt(n2)


def n_glass(wl_m, material='stih6'):
    """Dispatcher for glass materials (extensible)."""
    if material == 'stih6':
        return n_stih6(wl_m)
    raise ValueError(f"Unknown glass material: {material}")


def group_index(wl_m, material='stih6', delta=1e-12):
    """Group index  n_g = n − λ · dn/dλ  (numerical central difference, δλ=1 pm).

    For normal dispersion (dn/dλ < 0), the term −λ·dn/dλ > 0, so n_g > n.
    Because |dn/dλ| grows at shorter wavelengths, n_g is larger for blue (short λ)
    than for red (long λ).  This makes blue photons travel slower through glass,
    which is the physical origin of positive-chirp pulse stretching (see build_pulse).
    """
    n_hi = n_glass(wl_m + delta, material)
    n_lo = n_glass(wl_m - delta, material)
    dn_dlambda = (n_hi - n_lo) / (2 * delta)
    return n_glass(wl_m, material) - np.asarray(wl_m) * dn_dlambda


def group_delay(wl_m, glass_m, material='stih6'):
    """Group delay [s] for a spectral component at wavelength wl_m through glass_m metres.

    Shorter λ (blue) → larger n_g → larger GD → arrives later through normally dispersive glass.
    Longer  λ (red)  → smaller n_g → smaller GD → arrives earlier.
    """
    return glass_m * group_index(wl_m, material) / c


# ─────────────────────────────────────────────
#  PULSE HELPERS
# ─────────────────────────────────────────────

def spectral_nm_to_tau(fwhm_spectral_nm, lambda0_nm):
    """Spectral intensity FWHM [nm] → field Gaussian parameter τ [s].
    Assumes transform-limited pulse: Δt_FWHM · Δν_FWHM = 2·ln2/π.
    """
    lambda0_m = lambda0_nm * 1e-9
    fwhm_m    = fwhm_spectral_nm * 1e-9
    delta_nu  = c * fwhm_m / lambda0_m**2           # Hz
    fwhm_t_s  = 2.0 * np.log(2.0) / (np.pi * delta_nu)
    return fwhm_t_s / np.sqrt(2.0 * np.log(2.0))   # field τ [s]


def temporal_fwhm_fs(fwhm_spectral_nm, lambda0_nm):
    """Convenience: transform-limited intensity FWHM [fs] from spectral FWHM [nm]."""
    tau = spectral_nm_to_tau(fwhm_spectral_nm, lambda0_nm)
    return tau * np.sqrt(2.0 * np.log(2.0)) * 1e15


def time_bandwidth_product(fwhm_spectral_nm, lambda0_nm):
    """TBP = Δt_FWHM[fs] × Δν_FWHM[THz] for a transform-limited Gaussian (≈ 0.441)."""
    fwhm_t_fs    = temporal_fwhm_fs(fwhm_spectral_nm, lambda0_nm)
    delta_nu_THz = c * (fwhm_spectral_nm * 1e-9) / (lambda0_nm * 1e-9)**2 * 1e-12
    tbp = fwhm_t_fs * delta_nu_THz * 1e-3   # fs × THz × 1e-3 = dimensionless
    return fwhm_t_fs, delta_nu_THz, tbp


def marginal_temporal_intensity(E_2d):
    """Spectrally integrated instantaneous intensity  I(t) = Σᵢ E²(λᵢ, t).

    Sums the per-wavelength intensity over the spectral axis (axis 0), giving the
    total power delivered at each time step.

    Why the marginal peak drops after chirping (even though each slice is unchanged)
    ---------------------------------------------------------------------------------
    Glass is a linear lossless medium: it only shifts the arrival time of each
    spectral slice, leaving every slice's peak field amplitude A[i] intact.

    *Before* propagation (transform-limited):
        All n_spec slices peak simultaneously at t = 0.
        I_marginal(0) = Σᵢ A[i]²  ← the FULL sum — maximum possible power.

    *After* propagation (chirped, gd_spread >> τ_E):
        Slice i peaks at t = gd_s[i]; slices no longer overlap.
        At t ≈ gd_s[k], only slice k contributes appreciably:
        I_marginal(gd_s[k]) ≈ A[k]²  ← power of ONE slice only.

    Peak ratio ≈ max_k A[k]² / Σᵢ A[i]² = 1 / n_eff << 1,
    where n_eff = Σᵢ A[i]² is the effective number of contributing slices.

    Analogy: n_spec musicians each play at full volume.  Before chirping they
    all play the SAME note at the same instant (full combined power).  After
    chirping each plays alone in sequence — the instantaneous "loudness" is
    that of a single musician, even though each one is just as loud as before.

    Energy is conserved: ∫ I_marginal dt is identical before and after propagation.

    Parameters
    ----------
    E_2d : ndarray, shape [n_spec, n_time]  — field amplitude (not intensity)

    Returns
    -------
    I_marginal : ndarray, shape [n_time]
    """
    return np.sum(E_2d**2, axis=0)


def _fwhm_ps(t_ps, y):
    """Intensity FWHM [ps] of a 1-D profile y(t_ps) at the half-maximum level.

    Uses linear interpolation to locate the two half-maximum crossings.
    Returns NaN if the profile does not cross the half-maximum on both sides.
    """
    half = y.max() / 2.0
    i_pk = np.argmax(y)
    # Left crossing: y rises to the peak — interpolate on the ascending segment
    seg_l = y[:i_pk + 1]
    t_l   = t_ps[:i_pk + 1]
    t_left = np.interp(half, seg_l, t_l) if seg_l.min() < half else np.nan
    # Right crossing: y falls from the peak — reverse so values are ascending for interp
    seg_r = y[i_pk:][::-1]
    t_r   = t_ps[i_pk:][::-1]
    t_right = np.interp(half, seg_r, t_r) if seg_r.min() < half else np.nan
    return t_right - t_left


def build_pulse(fwhm_spectral_nm, lambda0_nm, glass_mm,
                material='stih6',
                n_spec=N_SPEC, n_time=N_TIME):
    """
    Build and propagate a pulse component-by-component through glass.

    Parameters
    ----------
    fwhm_spectral_nm : float
        Spectral intensity FWHM [nm]  (transform-limited pulse assumed).
    lambda0_nm : float
        Central wavelength [nm].
    glass_mm : float
        Glass propagation length [mm].

    Returns a dict with:
      lambdas_nm   [n_spec]         — spectral grid (nm)
      wn_cm1       [n_spec]         — wavenumber grid (cm⁻¹)
      A            [n_spec]         — field amplitude spectral weights  (A² = intensity weights)
      tau_s                         — field Gaussian parameter τ_E (s)
      t_ps_init    [n_time]         — time axis for initial pulse (ps)
      E_init       [n_spec, n_time] — 2-D field amplitude before propagation
      t_ps_prop    [n_time]         — time axis after propagation (ps)
      E_prop       [n_spec, n_time] — 2-D field amplitude after propagation
      gd_ps        [n_spec]         — relative group delays (ps)

    For display: intensity = E_init**2 or E_prop**2 (done inside plot_pulse_3d).
    For convolutions: use E_prop directly — field amplitudes required.

    Chirp physics
    -------------
    S-TIH6 has **normal dispersion** at 800–1050 nm: refractive index n decreases
    with wavelength (dn/dλ < 0), giving positive group-velocity dispersion (GVD > 0).
    The group index n_g = n − λ·dn/dλ satisfies n_g > n and is larger for shorter λ.

    Consequence on group delays (gd_s, computed below):
      - Shorter λ (blue, higher ν) → larger n_g → larger GD → gd_s[λ < λ₀] > 0
        → Gaussian centre at t > 0  → blue component arrives LATER at the sample.
      - Longer  λ (red,  lower ν) → smaller n_g → smaller GD → gd_s[λ > λ₀] < 0
        → Gaussian centre at t < 0  → red  component arrives EARLIER at the sample.

    In the **time domain** (at the sample), ω(t) increases with t — this is
    conventionally called **positive chirp**.

    In a **spatial snapshot** of the pulse propagating through space, the picture
    is reversed: the leading edge (front, which has already travelled further) is the
    red component (it moves faster); the trailing edge (back) is blue.  Scanning the
    spatial picture from front to back, frequency decreases — sometimes called
    "negative chirp" in this spatial convention.  Both descriptions refer to the
    same physical situation; the code operates in the time domain.

    Chirp consequences:
      - **Pulse elongation**: the temporal extent grows from τ_TL to approximately
        gd_spread = max(gd_s) − min(gd_s) (spectral-focusing limit, gd_spread >> τ_E).
      - **Peak power reduction**: spectral components no longer all peak at t = 0.
        Energy is conserved (∫I_marginal dt unchanged), but instantaneous peak power
        drops roughly as τ_TL / gd_spread.  Use marginal_temporal_intensity() to
        compute and visualise this effect explicitly.
      - **Resolution improvement**: the chirp rate (ps/nm) sets the instantaneous
        pump–Stokes frequency difference, which determines the CARS Raman linewidth.
        More glass → more chirp → narrower CARS bandwidth (better resolution) at the
        cost of lower peak signal.
    """
    tau_s = spectral_nm_to_tau(fwhm_spectral_nm, lambda0_nm)

    # Field amplitude σ: A_field = exp(-0.5*(λ-λ0)²/σ_E²), FWHM_field = FWHM_I × √2.
    # A_field² gives intensity weights with FWHM = fwhm_spectral_nm (input intensity FWHM).
    # Derivation: A² has FWHM_I = 2·σ_E·√ln2  →  σ_E = FWHM_I / (2·√ln2).
    sigma_nm = fwhm_spectral_nm / (2.0 * np.sqrt(np.log(2.0)))   # field amplitude σ [nm]

    # Spectral grid: ±3.5σ_E around central wavelength (covers field to < 0.3 % of peak)
    lam_min = lambda0_nm - 3.5 * sigma_nm
    lam_max = lambda0_nm + 3.5 * sigma_nm
    lambdas_nm = np.linspace(lam_min, lam_max, n_spec)
    wn_cm1 = 1e7 / lambdas_nm                               # cm⁻¹

    # Field amplitude spectral weights — FWHM = fwhm_spectral_nm × √2 (field FWHM)
    A = np.exp(-0.5 * ((lambdas_nm - lambda0_nm) / sigma_nm)**2)
    A /= A.max()

    # Group delays for each spectral component (absolute, then made relative)
    glass_m = glass_mm * 1e-3
    lambdas_m = lambdas_nm * 1e-9
    gd_s_abs = group_delay(lambdas_m, glass_m, material)          # [s], absolute
    # Reference: group delay of the central wavelength (subtracted for relative timing)
    gd_ref = group_delay(lambda0_nm * 1e-9, glass_m, material)
    gd_s = gd_s_abs - gd_ref                                      # [s], relative to centre
    gd_ps = gd_s * 1e12

    # Shared time window: wide enough to contain the full group-delay spread.
    # Both the initial and propagated grids use this window so their plots are
    # directly comparable on the same ps axis.
    gd_spread = gd_s.max() - gd_s.min()
    gd_half   = max(abs(gd_s.max()), abs(gd_s.min()))   # largest one-sided delay (handles asymmetry)
    t_half    = (gd_half + 3.0 * tau_s) * 1.15           # signal extent + 3σ buffer + 15% margin

    # --- Initial (unchirped) field amplitude time grid ---  centred at t = 0
    t_s_init = np.linspace(-t_half, t_half, n_time)
    t_ps_init = t_s_init * 1e12

    E_init = np.zeros((n_spec, n_time))
    for i in range(n_spec):
        E_init[i, :] = A[i] * np.exp(-t_s_init**2 / tau_s**2)    # field: exp(-t²/τ_E²)

    # --- Propagated field amplitude: each spectral slice shifted by its group delay ---
    # gd_s[i] > 0 for blue (λ < λ₀): Gaussian peak is at t > 0 → blue arrives later.
    # gd_s[i] < 0 for red  (λ > λ₀): Gaussian peak is at t < 0 → red  arrives earlier.
    # The resulting time-domain field is positively chirped: ω increases with t.
    # Because the slices are temporally separated, they no longer all peak at t = 0,
    # so the total instantaneous power (marginal intensity) is lower than for the
    # transform-limited pulse — see marginal_temporal_intensity().
    t_s_prop = np.linspace(-t_half, t_half, n_time)
    t_ps_prop = t_s_prop * 1e12

    E_prop = np.zeros((n_spec, n_time))
    for i in range(n_spec):
        E_prop[i, :] = A[i] * np.exp(-(t_s_prop - gd_s[i])**2 / tau_s**2)  # field

    return dict(
        lambdas_nm=lambdas_nm,
        wn_cm1=wn_cm1,
        A=A,
        tau_s=tau_s,
        t_ps_init=t_ps_init,
        E_init=E_init,
        t_ps_prop=t_ps_prop,
        E_prop=E_prop,
        gd_ps=gd_ps,
        lambda0_nm=lambda0_nm,
        fwhm_spectral_nm=fwhm_spectral_nm,
        glass_mm=glass_mm,
        material=material,
    )


# ─────────────────────────────────────────────
#  CONVOLUTIONS
# ─────────────────────────────────────────────

def _interp_rows(E_src, t_src, t_dst):
    """Interpolate each spectral row of E_src (field amplitude, [n_spec, n_time])
    from time axis t_src onto t_dst.  Returns zeros outside the source range."""
    out = np.zeros((E_src.shape[0], len(t_dst)))
    for i in range(E_src.shape[0]):
        f = interp1d(t_src, E_src[i], bounds_error=False, fill_value=0.0)
        out[i] = f(t_dst)
    return out


def compute_conv1(pump, stokes, raman_axis, delay_ps=0.0):
    """
    Raman excitation amplitude  C₁(Ω, t).

        C₁(Ω, t) = Σₖ E_pump(νₖ, t) · E_Stokes(νₖ − Ω, t + delay_ps)

    E_pump and E_Stokes are field amplitudes stored directly in E_prop
    (no square root required — the arrays are not intensities).
    This is the physically correct driving term for the Raman coherence:
    ρ(Ω,t) ∝ E_pump(t) × E_Stokes*(t).  Using intensities (I × I) instead
    would square the coherence and artificially narrow its spectral width by √2.

    Implementation is fully vectorised:
    - E_Stokes is interpolated once onto a fine wavenumber grid
    - For each Raman shift Ω the pump wavenumber slice pump_wn − Ω is
      mapped to indices on that grid via a simple offset — no per-Ω loop.

    Parameters
    ----------
    pump, stokes : dicts from build_pulse
    raman_axis   : 1-D array of Raman shifts [cm⁻¹]
    delay_ps     : time offset applied to Stokes [ps]

    Returns
    -------
    t_common : 1-D time axis [ps]
    C1       : [n_raman, n_time]  — Raman excitation amplitude (field units)
    """
    # Pump: shorter λ → higher ν.  Stokes: longer λ → lower ν.
    # CARS condition: ν_Stokes = ν_pump − Ω  (standard convention)
    pump_wn   = pump['wn_cm1']    # [n_pump]   — higher ν side
    stokes_wn = stokes['wn_cm1']  # [n_stokes] — lower  ν side

    t_pump   = pump['t_ps_prop']
    t_stokes = stokes['t_ps_prop'] + delay_ps

    # ── common time axis ──────────────────────────────────────
    t_min = min(t_pump.min(), t_stokes.min())
    t_max = max(t_pump.max(), t_stokes.max())
    n_common = max(len(t_pump), len(t_stokes))
    t_common = np.linspace(t_min, t_max, n_common)

    # E_prop already stores field amplitudes — use directly (no sqrt needed)
    P = _interp_rows(pump['E_prop'],   t_pump,   t_common)   # [n_pump, n_t]   field amplitude
    S = _interp_rows(stokes['E_prop'], t_stokes, t_common)   # [n_stokes, n_t] field amplitude

    # ── build a fine wavenumber grid covering pump_wn − Ω for all Ω ──
    # Standard CARS: Stokes at ν_pump − Ω  (Stokes is lower ν)
    wn_min = pump_wn.min() - raman_axis.max()
    wn_max = pump_wn.max() - raman_axis.min()
    d_wn   = (stokes_wn.max() - stokes_wn.min()) / (len(stokes_wn) - 1)
    n_fine = max(int((wn_max - wn_min) / d_wn) + 2, len(stokes_wn))
    wn_fine = np.linspace(wn_min - d_wn, wn_max + d_wn, n_fine)

    # Interpolate S onto the fine grid once  →  S_fine[n_fine, n_t]
    f_stokes = interp1d(stokes_wn, S, axis=0, bounds_error=False, fill_value=0.0)
    S_fine = f_stokes(wn_fine)                               # [n_fine, n_t]

    # For each pump wavenumber ν_k and Raman shift Ω we need S at ν_k − Ω.
    # shifted_wn[j, k] = pump_wn[k] − raman_axis[j]  →  shape [n_raman, n_pump]
    shifted_wn = pump_wn[np.newaxis, :] - raman_axis[:, np.newaxis]

    # Map shifted_wn to indices in wn_fine (linear index since wn_fine is uniform)
    idx = (shifted_wn - wn_fine[0]) / (wn_fine[1] - wn_fine[0])
    idx = np.clip(np.round(idx).astype(int), 0, n_fine - 1)

    # S_at_shift[j, k, t] = S_fine[idx[j,k], t]
    S_at_shift = S_fine[idx, :]                              # [n_raman, n_pump, n_t]

    # C1[j, t] = Σ_k P[k, t] · S_at_shift[j, k, t]
    C1 = np.einsum('kt,jkt->jt', P, S_at_shift)

    C1 = np.maximum(C1, 0.0)
    return t_common, C1


def find_optimal_delay(pump, stokes, raman_axis):
    """
    Scan Stokes time delays and return the one that maximises the total
    integrated C₁ signal — i.e., the delay of maximum temporal overlap.

    This is a signal-strength criterion, not a resolution criterion.
    It answers the question: at what delay do the two chirped pulses
    overlap the most in time, producing the strongest Raman excitation?

    For perfectly matched chirp rates the answer is δt ≈ 0 (pulses centred
    on each other).  For mismatched chirp rates (different glass lengths or
    different centre wavelengths) it may shift slightly.

    Note: maximising total C₁ signal does NOT guarantee minimum spectral
    width (best resolution).  The delay that minimises the RMS width of
    C₁(Ω) can differ, especially when pump and Stokes chirp rates are
    mismatched.  A coarser Raman grid is used here for speed.
    """
    delays  = np.linspace(-DELAY_SCAN_PS_RANGE, DELAY_SCAN_PS_RANGE, DELAY_SCAN_N)
    raman_c = np.linspace(raman_axis.min(), raman_axis.max(), max(N_RAMAN // 3, 20))
    signal_vals = np.zeros(len(delays))

    for k, d in enumerate(delays):
        _, C1 = compute_conv1(pump, stokes, raman_c, delay_ps=d)
        signal_vals[k] = C1.sum()

    best_idx = np.argmax(signal_vals)
    opt_d = delays[best_idx]

    # Compute stats at optimal delay for reporting
    _, C1_opt = compute_conv1(pump, stokes, raman_c, delay_ps=opt_d)
    nu_bar, sigma = spectral_stats(raman_c, C1_opt.sum(axis=1))
    print(f"  Optimal delay (max signal): δt = {opt_d:.3f} ps  "
          f"→ center = {nu_bar:.1f} cm⁻¹,  RMS = {sigma:.1f} cm⁻¹")
    return opt_d


def compute_conv2_2d(C1_2d, pump, t_common, raman_axis):
    """
    Time-resolved anti-Stokes excitation amplitude:
        C₂(ν_AS, t) = ∫ C₁(Ω, t) · E_pump(ν_AS − Ω, t) dΩ

    where E_pump is the pump field amplitude stored directly in E_prop
    (no square root — not an intensity), and C₁ is the Raman coherence
    amplitude from compute_conv1.  The full CARS field goes as
    E_pump × coherence, so the probe step is at field level throughout.

    Anti-Stokes axis: ν_AS = ν_pump_center + Ω  (~15 k cm⁻¹, ~654 nm)

    The pump kernel is time-dependent (different spectral slices arrive at different
    times due to chirp), which is the essence of spectral focusing.

    Returns
    -------
    as_axis : [n_raman]         anti-Stokes wavenumbers [cm⁻¹]
    C2_2d   : [n_raman, n_time] time-resolved anti-Stokes excitation amplitude
    """
    nu_pump_center = 1e7 / pump['lambda0_nm']
    as_axis = nu_pump_center + raman_axis

    # E_prop already stores field amplitudes — use directly (no sqrt needed)
    P = _interp_rows(pump['E_prop'], pump['t_ps_prop'], t_common)

    pump_wn = pump['wn_cm1']
    dOmega  = raman_axis[1] - raman_axis[0]

    # Precompute centred kernel axis (fixed for all time steps)
    pump_marginal = P.sum(axis=1)
    wn_center     = np.average(pump_wn, weights=pump_marginal + 1e-30)
    pump_shift    = pump_wn - wn_center
    n_half        = int(np.ceil((pump_shift.max() - pump_shift.min()) / dOmega / 2)) + 1
    pump_axis_c   = np.arange(-n_half, n_half + 1) * dOmega

    # Vectorised interpolation across all time steps at once  →  P_kern[n_kern, n_time]
    f_pump  = interp1d(pump_shift, P, axis=0, bounds_error=False, fill_value=0.0)
    P_kern  = np.maximum(f_pump(pump_axis_c), 0.0)
    # No normalisation: P_kern carries the actual pump field amplitude at each time step.
    # Chirping reduces the peak amplitude — this effect must propagate into C2.

    C2_2d = np.zeros_like(C1_2d)
    for ti in range(len(t_common)):
        C2_2d[:, ti] = np.maximum(
            fftconvolve(C1_2d[:, ti], P_kern[:, ti], mode='same'), 0.0)
    return as_axis, C2_2d


def spectral_stats(raman_axis, spectrum):
    """Centroid and RMS width of a 1-D spectrum on raman_axis."""
    norm = spectrum.sum()
    if norm == 0:
        return np.nan, np.nan
    nu_bar = np.dot(raman_axis, spectrum) / norm
    sigma  = np.sqrt(np.dot((raman_axis - nu_bar)**2, spectrum) / norm)
    return nu_bar, sigma


def run_delay_scan(pump, stokes, raman_axis):
    """
    Scan Stokes delay τ over [−DELAY_SCAN_PS_RANGE, +DELAY_SCAN_PS_RANGE] and
    record at each delay:

      p1 — total C₁ field amplitude  (∫∫ C₁ dΩ dt, consistent with find_optimal_delay)
      c1 — power-weighted Raman centre [cm⁻¹]
      p2 — total C₂ field amplitude  (∫∫ C₂ dν dt)
      c2 — power-weighted anti-Stokes centre [cm⁻¹]

    The anti-Stokes axis is delay-independent (depends only on pump centre λ and
    raman_axis), so it is pre-computed once and returned for use in plotting.

    Intended for Scenario A only (pump 15 cm + Stokes 15 cm).
    """
    delays = np.linspace(-DELAY_SCAN_PS_RANGE, DELAY_SCAN_PS_RANGE, DELAY_SCAN_N)
    p1 = np.zeros(len(delays))
    c1 = np.zeros(len(delays))
    p2 = np.zeros(len(delays))
    c2 = np.zeros(len(delays))
    eps = 1e-30

    # Anti-Stokes axis is delay-independent — pre-compute once
    nu_pump_center = 1e7 / pump['lambda0_nm']
    as_axis = nu_pump_center + raman_axis

    print(f"  Scanning {len(delays)} delays  "
          f"τ ∈ [{delays[0]:+.1f}, {delays[-1]:+.1f}] ps …")

    for k, tau in enumerate(delays):
        if k == 0 or (k + 1) % 30 == 0 or k == len(delays) - 1:
            print(f"    [{k+1:3d}/{len(delays)}]  τ = {tau:+.3f} ps")

        t_common, C1 = compute_conv1(pump, stokes, raman_axis, delay_ps=tau)
        p1[k] = C1.sum()
        c1[k] = np.average(raman_axis, weights=C1.sum(axis=1) + eps)

        _, C2 = compute_conv2_2d(C1, pump, t_common, raman_axis)
        p2[k] = C2.sum()
        c2[k] = np.average(as_axis, weights=C2.sum(axis=1) + eps)

    # Print summary statistics
    def _scan_stats(power, center, name, axis_unit='cm⁻¹'):
        peak_val  = power.max()
        tau_mean  = np.average(delays, weights=power)
        tau_rms   = np.sqrt(np.average((delays - tau_mean)**2, weights=power))
        mask      = power > 0.01 * peak_val
        if mask.any():
            c_min, c_max = center[mask].min(), center[mask].max()
        else:
            c_min = c_max = np.nan
        print(f"  {name}: peak at τ = {delays[np.argmax(power)]:+.3f} ps  |  "
              f"centroid = {tau_mean:+.3f} ps  |  RMS width = {tau_rms:.3f} ps")
        print(f"      accessible {axis_unit} range: {c_min:.0f} – {c_max:.0f}  "
              f"(span = {c_max - c_min:.0f} {axis_unit})")

    print()
    _scan_stats(p1, c1, 'C₁ (Raman excitation)', axis_unit='cm⁻¹')
    _scan_stats(p2, c2, 'C₂ (anti-Stokes signal)', axis_unit='cm⁻¹')

    return delays, p1, c1, p2, c2, as_axis


# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────

def _save(fig, step_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{step_name}.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"  saved → {path}")


def _save_plotly(pfig, step_name, output_dir):
    if not _PLOTLY:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{step_name}.html')
    pfig.write_html(path)
    print(f"  saved → {path}")


def plot_pulse_3d(t_ps, lambdas_nm, E_2d, title, step_name, output_dir):
    """2-D heatmap (PDF) + 3-D surface (HTML): time × wavelength × intensity.

    The PDF uses pcolormesh (2-D) to avoid matplotlib surface aliasing that
    occurs when the pulse peak spans only a few time samples in the large
    time window (e.g. an 80 fs TL pulse in a 6 ps window).  The interactive
    HTML retains the full 3-D surface rendered by plotly from all data points.

    E_2d is field amplitude [n_spec, n_time]; intensity = E_2d² is computed here.
    """
    I_2d = E_2d**2

    # ── matplotlib: 2-D heatmap (PDF) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    pc = ax.pcolormesh(t_ps, lambdas_nm, I_2d, cmap='viridis', shading='auto')
    fig.colorbar(pc, ax=ax, label='Intensity  E²')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Wavelength (nm)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)

    # ── plotly: 3-D surface (HTML, interactive) ─────────────────────────────────
    if _PLOTLY:
        T, L = np.meshgrid(t_ps, lambdas_nm)
        pfig = go.Figure(data=[go.Surface(x=T, y=L, z=I_2d,
                                          colorscale='Viridis', opacity=0.9)])
        pfig.update_layout(
            title=title,
            scene=dict(xaxis_title='Time (ps)',
                       yaxis_title='Wavelength (nm)',
                       zaxis_title='Intensity I'),
            width=900, height=650)
        _save_plotly(pfig, step_name, output_dir)

    return fig


def plot_marginal_comparison(pulse, beam_label, step_name, output_dir):
    """Marginal temporal intensity I(t) = Σᵢ E²(λᵢ, t) — initial vs. chirped.

    Both profiles share the same absolute-amplitude axis so the peak-power
    reduction caused by glass propagation is visually unambiguous.  Annotated
    with the peak ratio and temporal FWHM of each profile.

    Parameters
    ----------
    pulse      : dict returned by build_pulse()
    beam_label : str, e.g. 'Pump' or 'Stokes'
    step_name  : str   output filename stem (no extension)
    output_dir : str   output directory
    """
    t_ps   = pulse['t_ps_init']   # same time window for both (shared grid)
    I_init = marginal_temporal_intensity(pulse['E_init'])
    I_prop = marginal_temporal_intensity(pulse['E_prop'])

    peak_init  = I_init.max()
    peak_prop  = I_prop.max()
    peak_ratio = peak_prop / peak_init
    fwhm_init  = _fwhm_ps(t_ps, I_init)
    fwhm_prop  = _fwhm_ps(t_ps, I_prop)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_ps, I_init, lw=2, color='steelblue',
            label=f'Initial (transform-limited)  FWHM = {fwhm_init:.3f} ps')
    ax.plot(t_ps, I_prop, lw=2, color='darkorange',
            label=f'After glass (chirped)         FWHM = {fwhm_prop:.2f} ps')

    # Annotate the peak-power reduction
    annotation = (f'Peak ratio: {peak_ratio:.4f}×\n'
                  f'(peak power reduced by {(1 - peak_ratio) * 100:.1f} %)\n'
                  f'Energy conserved: area unchanged')
    ax.annotate(annotation,
                xy=(0.62, 0.72), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.85))

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(r'Marginal intensity  $\Sigma_i\,E^2(\lambda_i,\,t)$  [arb. units]')
    ax.set_title(f'{beam_label} — marginal temporal intensity: initial vs. chirped\n'
                 f'Glass: {pulse["glass_mm"]} mm {pulse["material"].upper()}  '
                 f'| λ₀ = {pulse["lambda0_nm"]:.1f} nm')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)

    if _PLOTLY:
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(
            x=t_ps, y=I_init, mode='lines', name=f'Initial (TL)  FWHM={fwhm_init:.3f} ps',
            line=dict(color='steelblue', width=2)))
        pfig.add_trace(go.Scatter(
            x=t_ps, y=I_prop, mode='lines',
            name=f'After glass (chirped)  FWHM={fwhm_prop:.2f} ps',
            line=dict(color='darkorange', width=2)))
        pfig.update_layout(
            title=f'{beam_label} — marginal temporal intensity (initial vs. chirped)',
            xaxis_title='Time (ps)',
            yaxis_title='Marginal intensity (arb. units)',
            width=850, height=500)
        _save_plotly(pfig, step_name, output_dir)

    return fig


def plot_conv1_3d(t_ps, raman_axis, C1_2d, title, step_name, output_dir):
    """3-D surface: time × Raman shift × excitation."""
    T, R = np.meshgrid(t_ps, raman_axis)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, R, C1_2d, cmap='inferno', alpha=0.9,
                    rcount=60, ccount=60)
    ax.set_xlabel('Time (ps)', labelpad=8)
    ax.set_ylabel('Raman shift (cm⁻¹)', labelpad=8)
    ax.set_zlabel('Excitation amplitude', labelpad=8)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)

    if _PLOTLY:
        pfig = go.Figure(data=[go.Surface(x=T, y=R, z=C1_2d,
                                          colorscale='Hot', opacity=0.9)])
        pfig.update_layout(
            title=title,
            scene=dict(xaxis_title='Time (ps)',
                       yaxis_title='Raman shift (cm⁻¹)',
                       zaxis_title='Excitation amplitude'),
            width=900, height=650)
        _save_plotly(pfig, step_name, output_dir)

    return fig


def plot_conv2_3d(t_ps, as_axis, C2_2d, title, step_name, output_dir):
    """3-D surface: time × anti-Stokes wavenumber × signal. PDF + interactive HTML.
    Y-axis shows cm⁻¹ with nm values added to tick labels.
    """
    T, AS = np.meshgrid(t_ps, as_axis)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, AS, C2_2d, cmap='plasma', alpha=0.9, rcount=60, ccount=60)
    ax.set_xlabel('Time (ps)', labelpad=8)
    ax.set_ylabel('Anti-Stokes (cm⁻¹)', labelpad=8)
    ax.set_zlabel('AS amplitude', labelpad=8)

    # Dual cm⁻¹/nm tick labels on y-axis (FixedLocator required before set_yticklabels)
    yticks = [v for v in ax.get_yticks() if as_axis.min() <= v <= as_axis.max()]
    ax.set_yticks(yticks)
    ytick_labels = [f'{v:.0f}\n({1e7/v:.0f} nm)' for v in yticks]
    ax.set_yticklabels(ytick_labels, fontsize=7)

    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)

    if _PLOTLY:
        nm_vals = 1e7 / as_axis
        nm_grid = np.tile(nm_vals[:, np.newaxis], (1, len(t_ps)))
        pfig = go.Figure(data=[go.Surface(
            x=T, y=AS, z=C2_2d,
            customdata=nm_grid,
            hovertemplate=(
                'Time: %{x:.2f} ps<br>'
                'ν_AS: %{y:.0f} cm⁻¹  (%{customdata:.1f} nm)<br>'
                'Signal: %{z:.3f}<extra></extra>'),
            colorscale='Plasma', opacity=0.9)])
        pfig.update_layout(
            title=title,
            scene=dict(xaxis_title='Time (ps)',
                       yaxis_title='Anti-Stokes (cm⁻¹)',
                       zaxis_title='AS amplitude'),
            width=900, height=650)
        _save_plotly(pfig, step_name, output_dir)
    return fig


def plot_projection(raman_axis, C2, nu_bar, sigma_rms, title, step_name, output_dir,
                    xlabel='Raman shift (cm⁻¹)'):
    """1-D spectral projection with RMS annotation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(raman_axis, C2, lw=2, color='steelblue')
    ax.axvline(nu_bar, color='red', lw=1.2, ls='--',
               label=f'Center: {nu_bar:.1f} cm⁻¹')
    ax.axvspan(nu_bar - sigma_rms, nu_bar + sigma_rms,
               alpha=0.15, color='red',
               label=f'RMS width: {sigma_rms:.1f} cm⁻¹')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)
    return fig


def plot_comparison(axis,
                    C_zero, nu_zero, sig_zero,
                    C_opt,  nu_opt,  sig_opt,
                    output_dir,
                    step_name='comparison',
                    xlabel='Shift − centroid (cm⁻¹)',
                    suptitle='Spectral resolution — centred comparison'):
    """Side-by-side spectral projections centred at their respective centroids."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    datasets = [
        (C_zero, nu_zero, sig_zero, 'Zero delay',    'steelblue',  axes[0]),
        (C_opt,  nu_opt,  sig_opt,  'Optimal delay', 'darkorange', axes[1]),
    ]
    for C, nu_bar, sigma, label, color, ax in datasets:
        ax.plot(axis - nu_bar, C, lw=2, color=color, label=label)
        ax.axvline(0, color='red', lw=1.0, ls='--')
        ax.axvspan(-sigma, sigma, alpha=0.15, color='red',
                   label=f'RMS: {sigma:.1f} cm⁻¹\nCenter: {nu_bar:.1f} cm⁻¹')
        ax.set_xlabel(xlabel)
        ax.set_title(label)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Amplitude')
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)
    return fig


def plot_delay_scan(delays, power, center,
                    title, center_label, step_name, output_dir):
    """
    Two-row figure for the Stokes-delay scan results (Scenario A only).

    Row 1 — Signal power vs. Stokes delay τ
        Power is normalised to its peak value.  A shaded fill-under curve
        emphasises the accessible delay range.  Key metrics (peak delay,
        centroid, RMS width of the curve) are annotated directly on the panel.

    Row 2 — Power-weighted spectral centre vs. Stokes delay τ
        Only the portion where power exceeds 1 % of the peak is shown as a
        scatter coloured by normalised power (plasma colourmap).  The
        accessible spectral range and tunable span are annotated directly
        on the panel.

    Parameters
    ----------
    delays       : [N]   Stokes delay axis [ps]
    power        : [N]   integrated signal at each delay (field-amplitude sum)
    center       : [N]   power-weighted spectral centre at each delay [cm⁻¹]
    title        : str   figure suptitle
    center_label : str   y-axis label for the spectral-centre panel
    step_name    : str   output filename stem (no extension)
    output_dir   : str   output directory
    """
    eps = 1e-30

    # ── Derived metrics ──────────────────────────────────────────────────────
    peak_val = power.max()
    peak_idx = np.argmax(power)
    tau_peak = delays[peak_idx]
    tau_mean = np.average(delays, weights=power + eps)
    tau_rms  = np.sqrt(np.average((delays - tau_mean)**2, weights=power + eps))

    mask = power > 0.01 * peak_val
    c_min  = center[mask].min()  if mask.any() else np.nan
    c_max  = center[mask].max()  if mask.any() else np.nan
    c_span = c_max - c_min       if mask.any() else np.nan
    c_mean = np.average(center[mask], weights=power[mask] + eps) if mask.any() else np.nan
    c_rms  = (np.sqrt(np.average((center[mask] - c_mean)**2, weights=power[mask] + eps))
              if mask.any() else np.nan)

    pn = power / (peak_val + eps)   # normalised for display

    # ── Matplotlib ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                   sharex=True, layout='constrained',
                                   gridspec_kw={'hspace': 0.10})

    # Panel 1 — power ─────────────────────────────────────────────────────────
    ax1.fill_between(delays, pn, alpha=0.22, color='steelblue')
    ax1.plot(delays, pn, lw=2, color='steelblue', label='Normalised signal power')
    ax1.axvline(tau_peak, color='crimson', lw=1.4, ls='--',
                label=f'Peak  τ = {tau_peak:+.3f} ps')
    ax1.axvline(tau_mean, color='darkorange', lw=1.1, ls=':',
                label=f'Centroid  τ = {tau_mean:+.3f} ps')

    info1 = (f'Peak:     τ = {tau_peak:+.3f} ps\n'
             f'Centroid: τ = {tau_mean:+.3f} ps\n'
             f'RMS width: {tau_rms:.3f} ps')
    ax1.annotate(info1, xy=(0.02, 0.97), xycoords='axes fraction',
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.45', fc='lightyellow', alpha=0.92))

    ax1.set_ylabel('Signal (normalised)', fontsize=10)
    ax1.set_ylim(-0.05, 1.20)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # Panel 2 — spectral centre ───────────────────────────────────────────────
    # Derive unit and numeric format from the center_label, e.g. "... (nm)" → "nm"
    _lp, _rp = center_label.rfind('('), center_label.rfind(')')
    center_unit = center_label[_lp + 1 : _rp] if _lp >= 0 and _rp > _lp else ''
    c_fmt = '.1f' if center_unit == 'nm' else '.0f'

    # Only plot the thresholded region — noisy tail values are omitted
    # Thresholded scatter coloured by normalised power
    if mask.any():
        sc = ax2.scatter(delays[mask], center[mask],
                         c=pn[mask], cmap='plasma',
                         s=12, zorder=3, vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(sc, ax=ax2, fraction=0.030, pad=0.015)
        cbar.set_label('Normalised power', fontsize=8)

        # Accessible range shading and bounds
        ax2.axhspan(c_min, c_max, alpha=0.08, color='green', zorder=0)
        ax2.axhline(c_min, color='seagreen', lw=0.9, ls='--', alpha=0.75,
                    label=f'{c_min:{c_fmt}} {center_unit}')
        ax2.axhline(c_max, color='seagreen', lw=0.9, ls='--', alpha=0.75,
                    label=f'{c_max:{c_fmt}} {center_unit}')

    info2 = (f'Accessible range: {c_min:{c_fmt}} – {c_max:{c_fmt}} {center_unit}\n'
             f'Tunable span: {c_span:{c_fmt}} {center_unit}\n'
             f'Power-weighted RMS: {c_rms:.2f} {center_unit}\n'
             f'(region where power > 1 % of peak)')
    ax2.annotate(info2, xy=(0.02, 0.97), xycoords='axes fraction',
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.45', fc='lightcyan', alpha=0.92))

    ax2.set_xlabel('Stokes delay  τ (ps)', fontsize=10)
    ax2.set_ylabel(center_label, fontsize=10)
    ax2.margins(y=0.18)
    ax2.grid(True, alpha=0.25)
    if mask.any():
        ax2.legend(fontsize=8, loc='upper right', title='Bounds (1 % threshold)')

    fig.suptitle(title, fontsize=11)
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)

    # ── Plotly HTML ───────────────────────────────────────────────────────────
    if _PLOTLY:
        from plotly.subplots import make_subplots as _make_subplots
        pfig = _make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=['Signal power (normalised)',
                            center_label + ' vs. Stokes delay'],
            vertical_spacing=0.10)

        # Row 1: power
        pfig.add_trace(go.Scatter(
            x=delays, y=pn, mode='lines', fill='tozeroy',
            line=dict(color='steelblue', width=2),
            name='Power (normalised)'), row=1, col=1)
        pfig.add_trace(go.Scatter(
            x=[tau_peak], y=[pn[peak_idx]], mode='markers',
            marker=dict(color='crimson', size=11, symbol='diamond'),
            name=f'Peak  τ={tau_peak:+.3f} ps'), row=1, col=1)
        pfig.add_trace(go.Scatter(
            x=[tau_mean, tau_mean], y=[0, 1], mode='lines',
            line=dict(color='darkorange', width=1.5, dash='dot'),
            name=f'Centroid τ={tau_mean:+.3f} ps'), row=1, col=1)

        # Row 2: spectral centre (thresholded, coloured by power)
        if mask.any():
            pfig.add_trace(go.Scatter(
                x=delays[mask], y=center[mask], mode='markers',
                marker=dict(color=pn[mask], colorscale='Plasma',
                            size=5, showscale=True,
                            colorbar=dict(title='Norm. power',
                                          len=0.42, y=0.20)),
                name='Spectral centre (power > 1 %)'), row=2, col=1)

        pfig.update_layout(
            title=title,
            xaxis2_title='Stokes delay  τ (ps)',
            yaxis_title='Signal (normalised)',
            yaxis2_title=center_label,
            width=860, height=720)
        _save_plotly(pfig, step_name, output_dir)

    return fig


# ─────────────────────────────────────────────
#  SCENARIO PIPELINE
# ─────────────────────────────────────────────

def run_scenario(pump_glass_mm, stokes_glass_mm, output_dir, scenario_label):
    """
    Run the full 10-step chirping + CARS convolution pipeline for one
    combination of glass lengths.  All plots are saved under output_dir.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {scenario_label}")
    print(f"  Pump {PUMP_LAMBDA0_NM} nm · {PUMP_FWHM_NM} nm FWHM · {pump_glass_mm} mm {GLASS_MATERIAL}")
    print(f"  Stokes {STOKES_LAMBDA0_NM} nm · {STOKES_FWHM_NM} nm FWHM · {stokes_glass_mm} mm {GLASS_MATERIAL}")
    print(sep)

    raman_axis = np.linspace(RAMAN_MIN_CM1, RAMAN_MAX_CM1, N_RAMAN)

    # ── [1] Build pump ────────────────────────────────────────
    print(f"\n[1/10] Building pump …")
    pump = build_pulse(PUMP_FWHM_NM, PUMP_LAMBDA0_NM, pump_glass_mm,
                       GLASS_MATERIAL, N_SPEC, N_TIME)
    p_fwhm_fs, p_dnu_THz, p_tbp = time_bandwidth_product(PUMP_FWHM_NM, PUMP_LAMBDA0_NM)
    p_gd_spread = pump['gd_ps'].max() - pump['gd_ps'].min()
    p_peak_ratio = (marginal_temporal_intensity(pump['E_prop']).max() /
                    marginal_temporal_intensity(pump['E_init']).max())
    print(f"  τ_TL = {p_fwhm_fs:.1f} fs  Δν = {p_dnu_THz:.2f} THz  TBP = {p_tbp:.3f}  "
          f"GD spread = {p_gd_spread:.3f} ps")
    print(f"  Chirp effect: peak power ratio (prop/init) = {p_peak_ratio:.4f}× "
          f"(−{(1 - p_peak_ratio) * 100:.1f} %)  |  elongation ≈ {p_gd_spread * 1e3 / p_fwhm_fs:.0f}× TL")

    # ── [2] Build Stokes ──────────────────────────────────────
    print(f"\n[2/10] Building Stokes …")
    stokes = build_pulse(STOKES_FWHM_NM, STOKES_LAMBDA0_NM, stokes_glass_mm,
                         GLASS_MATERIAL, N_SPEC, N_TIME)
    s_fwhm_fs, s_dnu_THz, s_tbp = time_bandwidth_product(STOKES_FWHM_NM, STOKES_LAMBDA0_NM)
    s_gd_spread = stokes['gd_ps'].max() - stokes['gd_ps'].min()
    s_peak_ratio = (marginal_temporal_intensity(stokes['E_prop']).max() /
                    marginal_temporal_intensity(stokes['E_init']).max())
    print(f"  τ_TL = {s_fwhm_fs:.1f} fs  Δν = {s_dnu_THz:.2f} THz  TBP = {s_tbp:.3f}  "
          f"GD spread = {s_gd_spread:.3f} ps")
    print(f"  Chirp effect: peak power ratio (prop/init) = {s_peak_ratio:.4f}× "
          f"(−{(1 - s_peak_ratio) * 100:.1f} %)  |  elongation ≈ {s_gd_spread * 1e3 / s_fwhm_fs:.0f}× TL")

    # ── [3] Marginal intensity: initial vs. chirped ──────────
    print("\n[3/10] Marginal temporal intensity comparison (chirp effect on peak power) …")
    plot_marginal_comparison(pump,   'Pump',   'pump_step1b_marginal',   output_dir)
    plot_marginal_comparison(stokes, 'Stokes', 'stokes_step1b_marginal', output_dir)

    # ── [4] Plot initial pulses ───────────────────────────────
    print("\n[4/10] 3-D plots: initial (unchirped) pulses …")
    plot_pulse_3d(pump['t_ps_init'],   pump['lambdas_nm'],   pump['E_init'],
                  f'Pump — initial  (λ₀={PUMP_LAMBDA0_NM} nm, Δλ={PUMP_FWHM_NM} nm, '
                  f'τ_TL={p_fwhm_fs:.0f} fs, TBP={p_tbp:.3f})',
                  'pump_step0_initial', output_dir=output_dir)

    plot_pulse_3d(stokes['t_ps_init'], stokes['lambdas_nm'], stokes['E_init'],
                  f'Stokes — initial  (λ₀={STOKES_LAMBDA0_NM} nm, Δλ={STOKES_FWHM_NM} nm, '
                  f'τ_TL={s_fwhm_fs:.0f} fs, TBP={s_tbp:.3f})',
                  'stokes_step0_initial', output_dir=output_dir)

    # ── [5] Plot propagated pulses ────────────────────────────
    print("\n[5/10] 3-D plots: after glass propagation …")
    plot_pulse_3d(pump['t_ps_prop'],   pump['lambdas_nm'],   pump['E_prop'],
                  f'Pump — after {pump_glass_mm} mm {GLASS_MATERIAL.upper()}',
                  'pump_step1_propagated', output_dir=output_dir)

    plot_pulse_3d(stokes['t_ps_prop'], stokes['lambdas_nm'], stokes['E_prop'],
                  f'Stokes — after {stokes_glass_mm} mm {GLASS_MATERIAL.upper()}',
                  'stokes_step1_propagated', output_dir=output_dir)

    # ── [6] Conv 1 — zero delay ───────────────────────────────
    print("\n[6/10] Raman excitation spectrum — zero delay …")
    t_zero, C1_zero = compute_conv1(pump, stokes, raman_axis, delay_ps=0.0)
    plot_conv1_3d(t_zero, raman_axis, C1_zero,
                  'Raman excitation  C₁(Ω, t) — zero delay',
                  'conv1_zero_delay', output_dir=output_dir)
    C1_zero_marg = C1_zero.sum(axis=1)

    # ── [7] Conv 1 — optimal delay ────────────────────────────
    print("\n[7/10] Scanning for optimal Stokes delay …")
    opt_delay_ps = find_optimal_delay(pump, stokes, raman_axis)
    t_opt, C1_opt = compute_conv1(pump, stokes, raman_axis, delay_ps=opt_delay_ps)
    plot_conv1_3d(t_opt, raman_axis, C1_opt,
                  f'Raman excitation  C₁(Ω, t) — optimal δt = {opt_delay_ps:+.3f} ps',
                  'conv1_optimal_delay', output_dir=output_dir)
    C1_opt_marg = C1_opt.sum(axis=1)

    # ── [8] C₁ comparison (Raman axis) ───────────────────────
    print("\n[8/10] C₁ comparison figure …")
    c1nu_zero, c1sig_zero = spectral_stats(raman_axis, C1_zero_marg)
    c1nu_opt,  c1sig_opt  = spectral_stats(raman_axis, C1_opt_marg)
    plot_comparison(raman_axis,
                    C1_zero_marg, c1nu_zero, c1sig_zero,
                    C1_opt_marg,  c1nu_opt,  c1sig_opt,
                    output_dir=output_dir,
                    step_name='comparison_c1',
                    xlabel='Raman shift − centroid (cm⁻¹)',
                    suptitle='C₁ — Raman excitation, centred comparison')

    # ── [9] Conv 2 — time-resolved anti-Stokes signal ─────────
    print("\n[9/10] Anti-Stokes signal C₂ (Conv 2) …")
    as_axis, C2_zero_2d = compute_conv2_2d(C1_zero, pump, t_zero, raman_axis)
    as_axis, C2_opt_2d  = compute_conv2_2d(C1_opt,  pump, t_opt,  raman_axis)

    plot_conv2_3d(t_zero, as_axis, C2_zero_2d,
                  'sf-CARS signal  C₂(ν_AS, t) — zero delay',
                  'conv2_zero_delay', output_dir=output_dir)
    plot_conv2_3d(t_opt,  as_axis, C2_opt_2d,
                  f'sf-CARS signal  C₂(ν_AS, t) — optimal δt = {opt_delay_ps:+.3f} ps',
                  'conv2_optimal_delay', output_dir=output_dir)

    C2_zero_marg = C2_zero_2d.sum(axis=1)
    C2_opt_marg  = C2_opt_2d.sum(axis=1)
    nu_zero, sig_zero = spectral_stats(as_axis, C2_zero_marg)
    nu_opt,  sig_opt  = spectral_stats(as_axis, C2_opt_marg)

    print(f"\n  ── RESULTS ──────────────────────────────────────")
    print(f"  Zero delay   : ν_AS = {nu_zero:.1f} cm⁻¹ ({1e7/nu_zero:.1f} nm),  "
          f"RMS = {sig_zero:.1f} cm⁻¹")
    print(f"  Optimal delay: ν_AS = {nu_opt:.1f} cm⁻¹ ({1e7/nu_opt:.1f} nm),  "
          f"RMS = {sig_opt:.1f} cm⁻¹  (δt = {opt_delay_ps:+.3f} ps)")

    plot_projection(as_axis, C2_zero_marg, nu_zero, sig_zero,
                    f'sf-CARS signal — zero delay\n'
                    f'ν_AS = {nu_zero:.1f} cm⁻¹ ({1e7/nu_zero:.1f} nm),  '
                    f'RMS = {sig_zero:.1f} cm⁻¹',
                    'conv2_zero_delay_projection', output_dir=output_dir,
                    xlabel='Anti-Stokes (cm⁻¹)')

    plot_projection(as_axis, C2_opt_marg, nu_opt, sig_opt,
                    f'sf-CARS signal — optimal delay (δt = {opt_delay_ps:+.3f} ps)\n'
                    f'ν_AS = {nu_opt:.1f} cm⁻¹ ({1e7/nu_opt:.1f} nm),  '
                    f'RMS = {sig_opt:.1f} cm⁻¹',
                    'conv2_optimal_delay_projection', output_dir=output_dir,
                    xlabel='Anti-Stokes (cm⁻¹)')

    # ── [10] C₂ comparison (anti-Stokes axis) ─────────────────
    print("\n[10/10] C₂ comparison figure …")
    plot_comparison(as_axis,
                    C2_zero_marg, nu_zero, sig_zero,
                    C2_opt_marg,  nu_opt,  sig_opt,
                    output_dir=output_dir,
                    step_name='comparison_c2',
                    xlabel='Anti-Stokes − centroid (cm⁻¹)',
                    suptitle='C₂ — sf-CARS signal, centred comparison')

    # ── [11–12] Delay scan — Scenario A only ─────────────────────────────────
    if pump_glass_mm == 150 and stokes_glass_mm == 150:
        print("\n[11/12] Delay scan — power & spectral centre vs Stokes delay …")
        delays, p1, c1, p2, c2, _ = run_delay_scan(pump, stokes, raman_axis)

        print("\n[12/12] Plotting delay-scan results …")
        plot_delay_scan(
            delays, p1, c1,
            title=(f'C₁ — Raman excitation power vs Stokes delay\n{scenario_label}'),
            center_label='Raman centre (cm⁻¹)',
            step_name='step11_c1_delay_scan',
            output_dir=output_dir)
        plot_delay_scan(
            delays, p2, 1e7 / c2,   # convert anti-Stokes centre cm⁻¹ → nm
            title=(f'C₂ — Anti-Stokes signal power vs Stokes delay\n{scenario_label}'),
            center_label='Anti-Stokes centre (nm)',
            step_name='step12_c2_delay_scan',
            output_dir=output_dir)

    print(f"\nOutputs saved in: {os.path.abspath(output_dir)}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    for pump_mm, stokes_mm, out_dir, label in SCENARIOS:
        run_scenario(pump_mm, stokes_mm, out_dir, label)
    plt.show()   # keep all windows open until closed by user


if __name__ == '__main__':
    main()
