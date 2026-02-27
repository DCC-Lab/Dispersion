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
    """Group index  n_g = n − λ · dn/dλ  (numerical central difference, δλ=1 pm)."""
    n_hi = n_glass(wl_m + delta, material)
    n_lo = n_glass(wl_m - delta, material)
    dn_dlambda = (n_hi - n_lo) / (2 * delta)
    return n_glass(wl_m, material) - np.asarray(wl_m) * dn_dlambda


def group_delay(wl_m, glass_m, material='stih6'):
    """Group delay [s] for a spectral component at wavelength wl_m through glass_m metres."""
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

    # --- Initial (unchirped) field amplitude time grid ---  centred at t = 0
    t_span_init = 8.0 * tau_s
    t_s_init = np.linspace(-t_span_init / 2, t_span_init / 2, n_time)
    t_ps_init = t_s_init * 1e12

    E_init = np.zeros((n_spec, n_time))
    for i in range(n_spec):
        E_init[i, :] = A[i] * np.exp(-t_s_init**2 / tau_s**2)    # field: exp(-t²/τ_E²)

    # --- Propagated field amplitude: each slice shifted by its group delay ---
    gd_spread = gd_s.max() - gd_s.min()
    t_half = max(gd_spread * 2.5, 5.0 * tau_s)
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
    """3-D surface: time × wavelength × intensity  (E_2d is field amplitude; squared here)."""
    I_2d = E_2d**2
    T, L = np.meshgrid(t_ps, lambdas_nm)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, L, I_2d, cmap='viridis', alpha=0.9,
                    rcount=60, ccount=60)
    ax.set_xlabel('Time (ps)', labelpad=8)
    ax.set_ylabel('Wavelength (nm)', labelpad=8)
    ax.set_zlabel('Intensity I', labelpad=8)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name, output_dir)

    if _PLOTLY:
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


# ─────────────────────────────────────────────
#  SCENARIO PIPELINE
# ─────────────────────────────────────────────

def run_scenario(pump_glass_mm, stokes_glass_mm, output_dir, scenario_label):
    """
    Run the full 8-step chirping + CARS convolution pipeline for one
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
    print(f"\n[1/9] Building pump …")
    pump = build_pulse(PUMP_FWHM_NM, PUMP_LAMBDA0_NM, pump_glass_mm,
                       GLASS_MATERIAL, N_SPEC, N_TIME)
    p_fwhm_fs, p_dnu_THz, p_tbp = time_bandwidth_product(PUMP_FWHM_NM, PUMP_LAMBDA0_NM)
    print(f"  τ_TL = {p_fwhm_fs:.1f} fs  Δν = {p_dnu_THz:.2f} THz  TBP = {p_tbp:.3f}  "
          f"GD spread = {pump['gd_ps'].max()-pump['gd_ps'].min():.3f} ps")

    # ── [2] Build Stokes ──────────────────────────────────────
    print(f"\n[2/9] Building Stokes …")
    stokes = build_pulse(STOKES_FWHM_NM, STOKES_LAMBDA0_NM, stokes_glass_mm,
                         GLASS_MATERIAL, N_SPEC, N_TIME)
    s_fwhm_fs, s_dnu_THz, s_tbp = time_bandwidth_product(STOKES_FWHM_NM, STOKES_LAMBDA0_NM)
    print(f"  τ_TL = {s_fwhm_fs:.1f} fs  Δν = {s_dnu_THz:.2f} THz  TBP = {s_tbp:.3f}  "
          f"GD spread = {stokes['gd_ps'].max()-stokes['gd_ps'].min():.3f} ps")

    # ── [3] Plot initial pulses ───────────────────────────────
    print("\n[3/9] 3-D plots: initial (unchirped) pulses …")
    plot_pulse_3d(pump['t_ps_init'],   pump['lambdas_nm'],   pump['E_init'],
                  f'Pump — initial  (λ₀={PUMP_LAMBDA0_NM} nm, Δλ={PUMP_FWHM_NM} nm, '
                  f'τ_TL={p_fwhm_fs:.0f} fs, TBP={p_tbp:.3f})',
                  'pump_step0_initial', output_dir=output_dir)

    plot_pulse_3d(stokes['t_ps_init'], stokes['lambdas_nm'], stokes['E_init'],
                  f'Stokes — initial  (λ₀={STOKES_LAMBDA0_NM} nm, Δλ={STOKES_FWHM_NM} nm, '
                  f'τ_TL={s_fwhm_fs:.0f} fs, TBP={s_tbp:.3f})',
                  'stokes_step0_initial', output_dir=output_dir)

    # ── [4] Plot propagated pulses ────────────────────────────
    print("\n[4/9] 3-D plots: after glass propagation …")
    plot_pulse_3d(pump['t_ps_prop'],   pump['lambdas_nm'],   pump['E_prop'],
                  f'Pump — after {pump_glass_mm} mm {GLASS_MATERIAL.upper()}',
                  'pump_step1_propagated', output_dir=output_dir)

    plot_pulse_3d(stokes['t_ps_prop'], stokes['lambdas_nm'], stokes['E_prop'],
                  f'Stokes — after {stokes_glass_mm} mm {GLASS_MATERIAL.upper()}',
                  'stokes_step1_propagated', output_dir=output_dir)

    # ── [5] Conv 1 — zero delay ───────────────────────────────
    print("\n[5/9] Raman excitation spectrum — zero delay …")
    t_zero, C1_zero = compute_conv1(pump, stokes, raman_axis, delay_ps=0.0)
    plot_conv1_3d(t_zero, raman_axis, C1_zero,
                  'Raman excitation  C₁(Ω, t) — zero delay',
                  'conv1_zero_delay', output_dir=output_dir)
    C1_zero_marg = C1_zero.sum(axis=1)

    # ── [6] Conv 1 — optimal delay ────────────────────────────
    print("\n[6/9] Scanning for optimal Stokes delay …")
    opt_delay_ps = find_optimal_delay(pump, stokes, raman_axis)
    t_opt, C1_opt = compute_conv1(pump, stokes, raman_axis, delay_ps=opt_delay_ps)
    plot_conv1_3d(t_opt, raman_axis, C1_opt,
                  f'Raman excitation  C₁(Ω, t) — optimal δt = {opt_delay_ps:+.3f} ps',
                  'conv1_optimal_delay', output_dir=output_dir)
    C1_opt_marg = C1_opt.sum(axis=1)

    # ── [7] C₁ comparison (Raman axis) ───────────────────────
    print("\n[7/9] C₁ comparison figure …")
    c1nu_zero, c1sig_zero = spectral_stats(raman_axis, C1_zero_marg)
    c1nu_opt,  c1sig_opt  = spectral_stats(raman_axis, C1_opt_marg)
    plot_comparison(raman_axis,
                    C1_zero_marg, c1nu_zero, c1sig_zero,
                    C1_opt_marg,  c1nu_opt,  c1sig_opt,
                    output_dir=output_dir,
                    step_name='comparison_c1',
                    xlabel='Raman shift − centroid (cm⁻¹)',
                    suptitle='C₁ — Raman excitation, centred comparison')

    # ── [8] Conv 2 — time-resolved anti-Stokes signal ─────────
    print("\n[8/9] Anti-Stokes signal C₂ (Conv 2) …")
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

    # ── [9] C₂ comparison (anti-Stokes axis) ─────────────────
    print("\n[9/9] C₂ comparison figure …")
    plot_comparison(as_axis,
                    C2_zero_marg, nu_zero, sig_zero,
                    C2_opt_marg,  nu_opt,  sig_opt,
                    output_dir=output_dir,
                    step_name='comparison_c2',
                    xlabel='Anti-Stokes − centroid (cm⁻¹)',
                    suptitle='C₂ — sf-CARS signal, centred comparison')

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
