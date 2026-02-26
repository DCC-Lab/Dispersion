"""
resolution.py — Spectral-focusing CARS resolution simulation

Step-by-step chirping of pump (1045 nm) and Stokes (805 nm) pulses through
S-TIH6 glass, followed by the pump⊗Stokes cross-correlation (Raman excitation)
and degenerate probe broadening, with spectral resolution readout.

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
PUMP_FWHM_FS      = 107.5      # Pump intensity FWHM [fs]
PUMP_LAMBDA0_NM   = 1040.85    # Pump central wavelength [nm]
PUMP_GLASS_MM     = 150.0      # Pump glass propagation length [mm]

STOKES_FWHM_FS    = 46.85      # Stokes intensity FWHM [fs]
STOKES_LAMBDA0_NM = 803.1577   # Stokes central wavelength [nm]
STOKES_GLASS_MM   = 100.0      # Stokes glass propagation length [mm]

GLASS_MATERIAL    = 'stih6'    # Currently only stih6 supported

N_SPEC   = 80     # Spectral grid points per beam
N_TIME   = 1000   # Time grid points

RAMAN_MIN_CM1 = 2500   # Raman axis lower bound [cm⁻¹]
RAMAN_MAX_CM1 = 3800   # Raman axis upper bound [cm⁻¹]
N_RAMAN       = 300    # Points on Raman axis

DELAY_SCAN_PS_RANGE = 8.0   # ± range for optimal-delay scan [ps]
DELAY_SCAN_N        = 150   # Number of delay points to scan

OUTPUT_DIR = 'plots/resolution'

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

def fwhm_to_tau(fwhm_fs):
    """Intensity FWHM [fs] → field Gaussian parameter τ [s].
    I(t) ∝ exp(−2t²/τ²),  FWHM_I = τ·√(2·ln 2).
    """
    return (fwhm_fs * 1e-15) / np.sqrt(2.0 * np.log(2.0))


def spectral_fwhm_nm(fwhm_fs, lambda0_nm):
    """Transform-limited spectral FWHM [nm] from temporal FWHM [fs]."""
    fwhm_s = fwhm_fs * 1e-15
    # Δt_FWHM · Δν_FWHM = 2·ln2/π  (transform limit for Gaussian)
    delta_nu = 2.0 * np.log(2.0) / (np.pi * fwhm_s)
    lambda0_m = lambda0_nm * 1e-9
    delta_lambda_m = (lambda0_m**2 / c) * delta_nu
    return delta_lambda_m * 1e9   # nm


def build_pulse(fwhm_fs, lambda0_nm, glass_mm,
                material='stih6',
                n_spec=N_SPEC, n_time=N_TIME):
    """
    Build and propagate a pulse component-by-component through glass.

    Returns a dict with:
      lambdas_nm   [n_spec]         — spectral grid (nm)
      wn_cm1       [n_spec]         — wavenumber grid (cm⁻¹)
      A            [n_spec]         — spectral Gaussian weights
      tau_s                         — field Gaussian parameter (s)
      t_ps_init    [n_time]         — time axis for initial pulse (ps)
      I_init       [n_spec, n_time] — 2-D intensity before propagation
      t_ps_prop    [n_time]         — time axis after propagation (ps)
      I_prop       [n_spec, n_time] — 2-D intensity after propagation
      gd_ps        [n_spec]         — group delays (ps)
    """
    tau_s   = fwhm_to_tau(fwhm_fs)
    sw_nm   = spectral_fwhm_nm(fwhm_fs, lambda0_nm)
    sigma_nm = sw_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # FWHM → 1-σ

    # Spectral grid: ±3.5σ around central wavelength
    lam_min = lambda0_nm - 3.5 * sigma_nm
    lam_max = lambda0_nm + 3.5 * sigma_nm
    lambdas_nm = np.linspace(lam_min, lam_max, n_spec)
    wn_cm1 = 1e7 / lambdas_nm                               # cm⁻¹

    # Spectral weights (Gaussian in wavelength)
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

    # --- Initial (unchirped) time grid ---  centred at t = 0
    t_span_init = 8.0 * tau_s
    t_s_init = np.linspace(-t_span_init / 2, t_span_init / 2, n_time)
    t_ps_init = t_s_init * 1e12

    I_init = np.zeros((n_spec, n_time))
    for i in range(n_spec):
        I_init[i, :] = A[i] * np.exp(-2.0 * t_s_init**2 / tau_s**2)

    # --- Propagated time grid: centred at t = 0,  wide enough for all gd-shifted slices ---
    gd_spread = gd_s.max() - gd_s.min()
    t_half = max(gd_spread * 2.5, 5.0 * tau_s)
    t_s_prop = np.linspace(-t_half, t_half, n_time)
    t_ps_prop = t_s_prop * 1e12

    I_prop = np.zeros((n_spec, n_time))
    for i in range(n_spec):
        I_prop[i, :] = A[i] * np.exp(-2.0 * (t_s_prop - gd_s[i])**2 / tau_s**2)

    return dict(
        lambdas_nm=lambdas_nm,
        wn_cm1=wn_cm1,
        A=A,
        tau_s=tau_s,
        t_ps_init=t_ps_init,
        I_init=I_init,
        t_ps_prop=t_ps_prop,
        I_prop=I_prop,
        gd_ps=gd_ps,
        lambda0_nm=lambda0_nm,
        fwhm_fs=fwhm_fs,
        glass_mm=glass_mm,
        material=material,
    )


# ─────────────────────────────────────────────
#  CONVOLUTIONS
# ─────────────────────────────────────────────

def _interp_rows(I_src, t_src, t_dst):
    """Interpolate each spectral row of I_src from t_src onto t_dst."""
    out = np.zeros((I_src.shape[0], len(t_dst)))
    for i in range(I_src.shape[0]):
        f = interp1d(t_src, I_src[i], bounds_error=False, fill_value=0.0)
        out[i] = f(t_dst)
    return out


def compute_conv1(pump, stokes, raman_axis, delay_ps=0.0):
    """
    Raman excitation spectrum  C₁(Ω, t).

        C₁(Ω, t) = Σₖ P(νₖ, t) · S_interp(νₖ + Ω, t + delay_ps)

    Implementation is fully vectorised:
    - S is interpolated once onto a fine wavenumber grid (n_fine points)
    - For each Raman shift Ω the pump wavenumber slice pump_wn + Ω is
      mapped to indices on that grid via a simple offset — no per-Ω loop.

    Parameters
    ----------
    pump, stokes : dicts from build_pulse
    raman_axis   : 1-D array of Raman shifts [cm⁻¹]
    delay_ps     : time offset applied to Stokes [ps]

    Returns
    -------
    t_common : 1-D time axis [ps]
    C1       : [n_raman, n_time]
    """
    pump_wn   = pump['wn_cm1']    # [n_pump]   ascending (short λ → high ν)
    stokes_wn = stokes['wn_cm1'] # [n_stokes]

    t_pump   = pump['t_ps_prop']
    t_stokes = stokes['t_ps_prop'] + delay_ps

    # ── common time axis ──────────────────────────────────────
    t_min = min(t_pump.min(), t_stokes.min())
    t_max = max(t_pump.max(), t_stokes.max())
    n_common = max(len(t_pump), len(t_stokes))
    t_common = np.linspace(t_min, t_max, n_common)

    P = _interp_rows(pump['I_prop'],   t_pump,   t_common)   # [n_pump, n_t]
    S = _interp_rows(stokes['I_prop'], t_stokes, t_common)   # [n_stokes, n_t]

    # ── build a fine wavenumber grid covering pump + all Raman shifts ──
    wn_min = pump_wn.min() + raman_axis.min()
    wn_max = pump_wn.max() + raman_axis.max()
    d_wn   = (stokes_wn.max() - stokes_wn.min()) / (len(stokes_wn) - 1)
    n_fine = max(int((wn_max - wn_min) / d_wn) + 2, len(stokes_wn))
    wn_fine = np.linspace(wn_min - d_wn, wn_max + d_wn, n_fine)

    # Interpolate S onto the fine grid once  →  S_fine[n_fine, n_t]
    f_stokes = interp1d(stokes_wn, S, axis=0, bounds_error=False, fill_value=0.0)
    S_fine = f_stokes(wn_fine)                               # [n_fine, n_t]

    # For each pump wavenumber ν_k and Raman shift Ω we need S at ν_k + Ω.
    # shifted_wn[j, k] = pump_wn[k] + raman_axis[j]  →  shape [n_raman, n_pump]
    shifted_wn = pump_wn[np.newaxis, :] + raman_axis[:, np.newaxis]

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

    For perfectly matched chirp rates this is δt ≈ 0.  For mismatched
    rates it may differ.  A coarser Raman grid is used for speed.
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
    print(f"  Optimal delay (max overlap): δt = {opt_d:.3f} ps  "
          f"→ center = {nu_bar:.1f} cm⁻¹,  RMS = {sigma:.1f} cm⁻¹")
    return opt_d


def compute_conv2(C1_marginal, pump_marginal_tuple, raman_axis):
    """
    Degenerate probe broadening:
        C₂(Ω) = C₁(Ω) ∗ P_pump(Ω)   (convolution)

    pump_marginal_tuple : (wn_axis, intensity) for the pump spectral marginal
    """
    pump_wn, pump_int = pump_marginal_tuple
    dOmega = raman_axis[1] - raman_axis[0]

    # Centre the pump kernel at 0
    wn_center = np.average(pump_wn, weights=pump_int + 1e-30)
    pump_shift = pump_wn - wn_center
    n_half = int(np.ceil((pump_shift.max() - pump_shift.min()) / dOmega / 2)) + 1
    pump_axis_c = np.arange(-n_half, n_half + 1) * dOmega

    f = interp1d(pump_shift, pump_int, bounds_error=False, fill_value=0.0)
    pump_kernel = np.maximum(f(pump_axis_c), 0.0)
    if pump_kernel.sum() > 0:
        pump_kernel /= pump_kernel.sum()

    C2 = np.maximum(fftconvolve(C1_marginal, pump_kernel, mode='same'), 0.0)
    return C2


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

def _save(fig, step_name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{step_name}.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"  saved → {path}")


def _save_plotly(pfig, step_name, output_dir=OUTPUT_DIR):
    if not _PLOTLY:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{step_name}.html')
    pfig.write_html(path)
    print(f"  saved → {path}")


def plot_pulse_3d(t_ps, lambdas_nm, I_2d, title, step_name):
    """3-D surface: time × wavelength × intensity."""
    I_norm = I_2d / (I_2d.max() or 1.0)
    T, L = np.meshgrid(t_ps, lambdas_nm)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, L, I_norm, cmap='viridis', alpha=0.9,
                    rcount=60, ccount=60)
    ax.set_xlabel('Time (ps)', labelpad=8)
    ax.set_ylabel('Wavelength (nm)', labelpad=8)
    ax.set_zlabel('Intensity (norm.)', labelpad=8)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name)

    if _PLOTLY:
        pfig = go.Figure(data=[go.Surface(x=T, y=L, z=I_norm,
                                          colorscale='Viridis', opacity=0.9)])
        pfig.update_layout(
            title=title,
            scene=dict(xaxis_title='Time (ps)',
                       yaxis_title='Wavelength (nm)',
                       zaxis_title='Intensity (norm.)'),
            width=900, height=650)
        _save_plotly(pfig, step_name)

    return fig


def plot_conv1_3d(t_ps, raman_axis, C1_2d, title, step_name):
    """3-D surface: time × Raman shift × excitation."""
    C1_norm = C1_2d / (C1_2d.max() or 1.0)
    T, R = np.meshgrid(t_ps, raman_axis)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, R, C1_norm, cmap='inferno', alpha=0.9,
                    rcount=60, ccount=60)
    ax.set_xlabel('Time (ps)', labelpad=8)
    ax.set_ylabel('Raman shift (cm⁻¹)', labelpad=8)
    ax.set_zlabel('Excitation (norm.)', labelpad=8)
    ax.set_title(title, pad=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name)

    if _PLOTLY:
        pfig = go.Figure(data=[go.Surface(x=T, y=R, z=C1_norm,
                                          colorscale='Hot', opacity=0.9)])
        pfig.update_layout(
            title=title,
            scene=dict(xaxis_title='Time (ps)',
                       yaxis_title='Raman shift (cm⁻¹)',
                       zaxis_title='Excitation (norm.)'),
            width=900, height=650)
        _save_plotly(pfig, step_name)

    return fig


def plot_projection(raman_axis, C2, nu_bar, sigma_rms, title, step_name):
    """1-D spectral projection with RMS annotation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(raman_axis, C2 / C2.max(), lw=2, color='steelblue')
    ax.axvline(nu_bar, color='red', lw=1.2, ls='--',
               label=f'Center: {nu_bar:.1f} cm⁻¹')
    ax.axvspan(nu_bar - sigma_rms, nu_bar + sigma_rms,
               alpha=0.15, color='red',
               label=f'RMS width: {sigma_rms:.1f} cm⁻¹')
    ax.set_xlabel('Raman shift (cm⁻¹)')
    ax.set_ylabel('Intensity (norm.)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, step_name)
    return fig


def plot_comparison(raman_axis,
                    C2_zero, nu_zero, sig_zero,
                    C2_opt,  nu_opt,  sig_opt,
                    output_dir=OUTPUT_DIR):
    """Side-by-side spectral projections centred at their respective centroids."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    datasets = [
        (C2_zero, nu_zero, sig_zero, 'Zero delay',    'steelblue',  axes[0]),
        (C2_opt,  nu_opt,  sig_opt,  'Optimal delay', 'darkorange', axes[1]),
    ]
    for C2, nu_bar, sigma, label, color, ax in datasets:
        ax.plot(raman_axis - nu_bar, C2 / C2.max(), lw=2, color=color, label=label)
        ax.axvline(0, color='red', lw=1.0, ls='--')
        ax.axvspan(-sigma, sigma, alpha=0.15, color='red',
                   label=f'RMS: {sigma:.1f} cm⁻¹\nCenter: {nu_bar:.1f} cm⁻¹')
        ax.set_xlabel('Raman shift − centroid (cm⁻¹)')
        ax.set_title(label)
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Intensity (norm.)')
    fig.suptitle('CARS spectral resolution — centred comparison', fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)
    _save(fig, 'comparison', output_dir)
    return fig


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Spectral-focusing CARS — resolution simulation")
    print("=" * 60)

    raman_axis = np.linspace(RAMAN_MIN_CM1, RAMAN_MAX_CM1, N_RAMAN)

    # ── [1] Build pump ────────────────────────────────────────
    print(f"\n[1/8] Pump  λ₀={PUMP_LAMBDA0_NM} nm  "
          f"FWHM={PUMP_FWHM_FS} fs  glass={PUMP_GLASS_MM} mm {GLASS_MATERIAL}")
    pump = build_pulse(PUMP_FWHM_FS, PUMP_LAMBDA0_NM, PUMP_GLASS_MM,
                       GLASS_MATERIAL, N_SPEC, N_TIME)
    print(f"  Spectral FWHM ≈ {spectral_fwhm_nm(PUMP_FWHM_FS, PUMP_LAMBDA0_NM):.3f} nm  "
          f"GD spread ≈ {pump['gd_ps'].max()-pump['gd_ps'].min():.3f} ps")

    # ── [2] Build Stokes ──────────────────────────────────────
    print(f"\n[2/8] Stokes  λ₀={STOKES_LAMBDA0_NM} nm  "
          f"FWHM={STOKES_FWHM_FS} fs  glass={STOKES_GLASS_MM} mm {GLASS_MATERIAL}")
    stokes = build_pulse(STOKES_FWHM_FS, STOKES_LAMBDA0_NM, STOKES_GLASS_MM,
                         GLASS_MATERIAL, N_SPEC, N_TIME)
    print(f"  Spectral FWHM ≈ {spectral_fwhm_nm(STOKES_FWHM_FS, STOKES_LAMBDA0_NM):.3f} nm  "
          f"GD spread ≈ {stokes['gd_ps'].max()-stokes['gd_ps'].min():.3f} ps")

    # ── [3] Plot initial pulses ───────────────────────────────
    print("\n[3/8] 3-D plots: initial (unchirped) pulses …")
    plot_pulse_3d(pump['t_ps_init'],   pump['lambdas_nm'],   pump['I_init'],
                  f'Pump — initial  (λ₀={PUMP_LAMBDA0_NM} nm, '
                  f'FWHM={PUMP_FWHM_FS} fs)',
                  'pump_step0_initial')

    plot_pulse_3d(stokes['t_ps_init'], stokes['lambdas_nm'], stokes['I_init'],
                  f'Stokes — initial  (λ₀={STOKES_LAMBDA0_NM} nm, '
                  f'FWHM={STOKES_FWHM_FS} fs)',
                  'stokes_step0_initial')

    # ── [4] Plot propagated pulses ────────────────────────────
    print("\n[4/8] 3-D plots: after glass propagation …")
    plot_pulse_3d(pump['t_ps_prop'],   pump['lambdas_nm'],   pump['I_prop'],
                  f'Pump — after {PUMP_GLASS_MM} mm {GLASS_MATERIAL.upper()}',
                  'pump_step1_propagated')

    plot_pulse_3d(stokes['t_ps_prop'], stokes['lambdas_nm'], stokes['I_prop'],
                  f'Stokes — after {STOKES_GLASS_MM} mm {GLASS_MATERIAL.upper()}',
                  'stokes_step1_propagated')

    # ── [5] Conv 1 — zero delay ───────────────────────────────
    print("\n[5/8] Raman excitation spectrum — zero delay …")
    t_zero, C1_zero = compute_conv1(pump, stokes, raman_axis, delay_ps=0.0)
    plot_conv1_3d(t_zero, raman_axis, C1_zero,
                  'Raman excitation  C₁(Ω, t) — zero delay',
                  'conv1_zero_delay')

    # ── [6] Conv 1 — optimal delay ────────────────────────────
    print("\n[6/8] Scanning for optimal Stokes delay …")
    opt_delay_ps = find_optimal_delay(pump, stokes, raman_axis)
    t_opt, C1_opt = compute_conv1(pump, stokes, raman_axis, delay_ps=opt_delay_ps)
    plot_conv1_3d(t_opt, raman_axis, C1_opt,
                  f'Raman excitation  C₁(Ω, t) — optimal δt = {opt_delay_ps:+.3f} ps',
                  'conv1_optimal_delay')

    # ── [7] Conv 2 — degenerate probe broadening ──────────────
    print("\n[7/8] Probe broadening (Conv 2) …")
    pump_marginal = (pump['wn_cm1'], pump['I_prop'].sum(axis=1))

    C2_zero = compute_conv2(C1_zero.sum(axis=1), pump_marginal, raman_axis)
    C2_opt  = compute_conv2(C1_opt.sum(axis=1),  pump_marginal, raman_axis)

    nu_zero, sig_zero = spectral_stats(raman_axis, C2_zero)
    nu_opt,  sig_opt  = spectral_stats(raman_axis, C2_opt)

    print(f"\n  ── RESULTS ──────────────────────────────────────")
    print(f"  Zero delay   : center = {nu_zero:.1f} cm⁻¹,  RMS = {sig_zero:.1f} cm⁻¹")
    print(f"  Optimal delay: center = {nu_opt:.1f} cm⁻¹,  RMS = {sig_opt:.1f} cm⁻¹  "
          f"(δt = {opt_delay_ps:+.3f} ps)")

    plot_projection(raman_axis, C2_zero, nu_zero, sig_zero,
                    f'CARS spectrum — zero delay\n'
                    f'center = {nu_zero:.1f} cm⁻¹,  RMS = {sig_zero:.1f} cm⁻¹',
                    'conv2_zero_delay_projection')

    plot_projection(raman_axis, C2_opt, nu_opt, sig_opt,
                    f'CARS spectrum — optimal delay (δt = {opt_delay_ps:+.3f} ps)\n'
                    f'center = {nu_opt:.1f} cm⁻¹,  RMS = {sig_opt:.1f} cm⁻¹',
                    'conv2_optimal_delay_projection')

    # ── [8] Comparison ────────────────────────────────────────
    print("\n[8/8] Comparison figure …")
    plot_comparison(raman_axis,
                    C2_zero, nu_zero, sig_zero,
                    C2_opt,  nu_opt,  sig_opt)

    print(f"\nAll outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
    plt.show()   # keep all windows open until closed by user


if __name__ == '__main__':
    main()
