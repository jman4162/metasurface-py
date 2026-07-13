"""Reproduce key results from Faenzi et al. 2019 (modulated MTS antennas).

Reference: M. Faenzi, G. Minatti, D. Gonzalez-Ovejero, F. Caminita,
E. Martini, C. Della Giovampaola, S. Maci, "Metasurface Antennas: New
Models, Applications and Realizations," Scientific Reports 9:10178, 2019.
DOI: 10.1038/s41598-019-46522-z

Reproduces (with the reduced-order adiabatic-Floquet-mode model):
- Fig. 5(a): broadside RHCP directivity pattern at 29.75 GHz, a = 13.5*lam,
  tapered aperture with taper*spillover ~= 0.85 (measured gain 37 dBi).
- Fig. 6: RHCP pencil beam tilted 30 deg at 20 GHz, a = 10*lam
  (measured co-pol directivity 33 dBi, aperture efficiency ~58%).
- Fig. 10-style maps: leaky-wave beta and alpha vs modulation index for
  the dual-band substrate (RO3010) at 26.25 and 32.05 GHz.
- Bandwidth: broadside directivity vs frequency for the Fig. 5 antenna,
  compared against the closed-form estimate B ~= 0.95*(v_g/c)/a_lambda.

Model scope: the paper's GR-MoM and FMM full-wave solvers, the printed
element synthesis, and the feed monopole are NOT modeled. The average
sheet reactance of the Fig. 5/6 prototypes is not quoted in the paper;
a typical value (beta_sw/k0 ~ 1.2) is assumed and stated on the figure.
Quoted reference metrics live in examples/data/faenzi2019_reference_metrics.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from metasurface_py.core.conventions import ETA_0, k0, wavelength
from metasurface_py.core.polarization import axial_ratio_db, circular_components
from metasurface_py.core.types import AngleGrid, SubstrateInfo
from metasurface_py.em.aperture_field import pattern_directivity
from metasurface_py.em.leakywave import (
    alpha_for_taper,
    dispersion_map,
    grounded_slab_reactance_tm,
    opaque_to_transparent,
    solve_sw_transparent,
    sw_wavenumber_tm,
)
from metasurface_py.plotting import save_figure, set_publication_style
from metasurface_py.surfaces.modulated import (
    ModulatedMetasurfaceAntenna,
    TensorModulation,
)

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"

FloatArray = npt.NDArray[np.floating]


def load_reference(figure: str) -> dict[str, float]:
    """Load quoted reference metrics for one figure of the paper."""
    out: dict[str, float] = {}
    with open(DATA_DIR / "faenzi2019_reference_metrics.csv") as f:
        for row in csv.DictReader(f):
            if row["figure"] == figure:
                out[row["quantity"]] = float(row["value"])
    return out


def sheet_reactance_for_x_op(
    x_op_target: float, freq: float, eps_r: float, thickness: float
) -> float:
    """Transparent sheet reactance that yields a target opaque reactance.

    The paper quotes design-level opaque reactances (e.g. 0.6*eta0) but
    not the cladding sheet reactance; invert the sheet <-> opaque mapping
    at the resulting surface-wave beta.
    """
    x_sheet = -300.0
    for _ in range(8):
        beta = sw_wavenumber_tm(x_op_target, freq)
        x_slab = grounded_slab_reactance_tm(freq, eps_r, thickness, beta)
        x_sheet = opaque_to_transparent(x_op_target, x_slab)
    return x_sheet


def tapered_rhcp_antenna(
    freq: float,
    radius_lam: float,
    x_op_target: float,
    substrate: SubstrateInfo,
    spillover_efficiency: float,
    n_rho: int,
    n_phi: int,
) -> ModulatedMetasurfaceAntenna:
    """Broadside RHCP antenna with m(rho) synthesized for a tapered aperture.

    Target amplitude: cosine on a 0.3 pedestal (edge taper ~ -10 dB), the
    classic high-efficiency illumination. alpha(rho) follows from the
    leaky-wave illumination-synthesis formula and is inverted to m(rho)
    through the dispersion map.
    """
    lam = wavelength(freq)
    radius = radius_lam * lam
    x_sheet = sheet_reactance_for_x_op(
        x_op_target, freq, substrate.eps_r, substrate.thickness_mm * 1e-3
    )
    sw = solve_sw_transparent(
        x_sheet, freq, substrate.eps_r, substrate.thickness_mm * 1e-3
    )

    rho = np.linspace(radius / n_rho, radius, n_rho)
    amplitude = 0.3 + 0.7 * np.cos(0.5 * np.pi * rho / radius)
    alpha_target = alpha_for_taper(rho, amplitude, efficiency=spillover_efficiency)

    period = 2.0 * np.pi / sw.beta
    m_grid = np.linspace(0.0, 0.45, 24)
    disp = dispersion_map(sw.x_op, m_grid, period, freq)
    alpha_grid = disp["alpha_over_k0"].values * k0(freq)
    m_of_rho = np.interp(alpha_target, alpha_grid, m_grid)

    def m_profile(r: FloatArray) -> FloatArray:
        return np.interp(r, rho, m_of_rho)

    modulation = TensorModulation.circular_broadside(
        sw.x_op, freq, m=m_profile, hand="rhcp"
    )
    return ModulatedMetasurfaceAntenna(
        radius=radius,
        x_transparent=x_sheet,
        substrate=substrate,
        modulation=modulation,
        n_rho=n_rho,
        n_phi=n_phi,
    )


def assembled_cut(
    pattern_db: FloatArray, theta: FloatArray, phi: FloatArray, phi_cut: float
) -> tuple[FloatArray, FloatArray]:
    """Assemble a +-theta cut from the phi_cut and phi_cut+180deg columns."""
    i_pos = int(np.argmin(np.abs(phi - phi_cut)))
    i_neg = int(np.argmin(np.abs(phi - (phi_cut + np.pi) % (2 * np.pi))))
    theta_deg = np.rad2deg(theta)
    x = np.concatenate([-theta_deg[::-1], theta_deg[1:]])
    y = np.concatenate([pattern_db[::-1, i_neg], pattern_db[1:, i_pos]])
    return x, y


def overlay_measured_if_present(ax: plt.Axes, name: str) -> None:
    """Overlay a digitized measured curve if the user has provided one."""
    path = DATA_DIR / name
    if path.exists():
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        ax.plot(data[:, 0], data[:, 1], "k.", ms=2, label="measured (digitized)")


class Validation:
    """Collect PASS/FAIL checks against the paper's quoted metrics."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool]] = []

    def check(self, name: str, detail: str, ok: bool) -> None:
        self.rows.append((name, detail, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def finish(self) -> None:
        failed = [r for r in self.rows if not r[2]]
        print(f"\n{len(self.rows) - len(failed)}/{len(self.rows)} checks passed")
        if failed:
            raise SystemExit(1)


def main() -> None:
    set_publication_style(target="ieee_double")
    FIG_DIR.mkdir(exist_ok=True)
    val = Validation()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ------------------------------------------------------------------
    # Panel A - Fig. 5(a): broadside RHCP pattern, 29.75 GHz, a = 13.5 lam
    # ------------------------------------------------------------------
    print("Panel A: Fig. 5(a) broadside RHCP antenna at 29.75 GHz ...")
    ref5 = load_reference("fig5a")
    f5 = ref5["frequency"] * 1e9
    ro3003 = SubstrateInfo(
        name="RO3003",
        eps_r=ref5["substrate_eps_r"],
        loss_tangent=ref5["substrate_loss_tangent"],
        thickness_mm=ref5["substrate_thickness"],
    )
    # Average reactance is not quoted for this prototype; assume a typical
    # modulated-MTS operating point beta_sw/k0 ~ 1.22 (x_op = 0.7*eta0).
    ant5 = tapered_rhcp_antenna(
        freq=f5,
        radius_lam=ref5["radius"],
        x_op_target=0.7 * ETA_0,
        substrate=ro3003,
        spillover_efficiency=0.90,
        n_rho=384,
        n_phi=192,
    )
    angles5 = AngleGrid.from_degrees(
        theta=np.linspace(0.0, 90.0, 241), phi=np.linspace(0.0, 355.0, 72)
    )
    ff5 = ant5.far_field(f5, angles5)
    d5 = pattern_directivity(ff5).values
    cp5 = circular_components(ff5["E_theta"], ff5["E_phi"])
    frac_r = np.abs(cp5["e_rhcp"].values) ** 2 / np.maximum(
        np.abs(cp5["e_rhcp"].values) ** 2 + np.abs(cp5["e_lhcp"].values) ** 2, 1e-300
    )
    d5_rhcp_db = 10 * np.log10(np.maximum(d5 * frac_r, 1e-12))
    d5_lhcp_db = 10 * np.log10(np.maximum(d5 * (1 - frac_r), 1e-12))

    it, ip = np.unravel_index(int(np.argmax(d5)), d5.shape)
    peak5_db = float(10 * np.log10(d5[it, ip]))
    bound5_db = float(
        10 * np.log10(4 * np.pi * ant5.aperture.area / wavelength(f5) ** 2)
    )
    ar5 = float(axial_ratio_db(ff5["E_theta"], ff5["E_phi"]).values[it, ip])

    ax = axes[0, 0]
    x5, y5 = assembled_cut(d5_rhcp_db, angles5.theta, angles5.phi, 0.0)
    x5x, y5x = assembled_cut(d5_lhcp_db, angles5.theta, angles5.phi, 0.0)
    ax.plot(x5, y5, label="AFM model, RHCP")
    ax.plot(x5x, y5x, "--", label="AFM model, LHCP")
    overlay_measured_if_present(ax, "faenzi2019_fig5a_measured.csv")
    ax.axhline(ref5["measured_gain"], color="0.4", lw=0.8, ls=":")
    ax.annotate("measured gain 37 dBi", (-88, ref5["measured_gain"] + 0.9), fontsize=7)
    ax.set(
        xlim=(-90, 90),
        ylim=(-20, 40),
        xlabel="Off-axis angle [deg]",
        ylabel="Directivity [dBi]",
        title="(a) Broadside RHCP, 29.75 GHz, a = 13.5$\\lambda$",
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.text(
        -88,
        -18,
        "assumed $\\beta_{sw}/k_0 = 1.22$ (not quoted in paper)",
        fontsize=6,
        color="0.35",
    )

    val.check(
        "Fig5a beam direction",
        f"theta_peak = {np.rad2deg(angles5.theta[it]):.2f} deg (expect 0)",
        abs(np.rad2deg(angles5.theta[it])) <= 0.5,
    )
    val.check(
        "Fig5a peak directivity",
        f"{peak5_db:.2f} dBi (paper: 37 dBi gain incl. losses; bound {bound5_db:.1f})",
        ref5["measured_gain"] - 0.5 <= peak5_db <= bound5_db,
    )
    val.check(
        "Fig5a axial ratio",
        f"{ar5:.2f} dB on boresight (paper: < {ref5['axial_ratio_max']} dB in HPBW)",
        ar5 < ref5["axial_ratio_max"],
    )

    # ------------------------------------------------------------------
    # Panel B - Fig. 6: RHCP beam tilted 30 deg, 20 GHz, a = 10 lam
    # ------------------------------------------------------------------
    print("Panel B: Fig. 6 tilted RHCP antenna at 20 GHz ...")
    ref6 = load_reference("fig6")
    f6 = ref6["frequency"] * 1e9
    # Substrate of this prototype is not quoted; assume the eps_r = 9.8,
    # h = 0.5 mm stack of the paper's isoflux prototype (Fig. 4).
    sub6 = SubstrateInfo(name="assumed", eps_r=9.8, thickness_mm=0.5)
    x_sheet6 = sheet_reactance_for_x_op(0.8 * ETA_0, f6, 9.8, 0.5e-3)
    sw6 = solve_sw_transparent(x_sheet6, f6, 9.8, 0.5e-3)
    mod6 = TensorModulation.circular_tilted(
        sw6.x_op, f6, m=0.3, theta0=np.deg2rad(ref6["beam_tilt"]), hand="rhcp"
    )
    ant6 = ModulatedMetasurfaceAntenna(
        radius=ref6["radius"] * wavelength(f6),
        x_transparent=x_sheet6,
        substrate=sub6,
        modulation=mod6,
        n_rho=320,
        n_phi=160,
    )
    angles6 = AngleGrid.from_degrees(
        theta=np.linspace(0.0, 90.0, 241), phi=np.linspace(0.0, 355.0, 72)
    )
    ff6 = ant6.far_field(f6, angles6)
    d6 = pattern_directivity(ff6).values
    cp6 = circular_components(ff6["E_theta"], ff6["E_phi"])
    frac_r6 = np.abs(cp6["e_rhcp"].values) ** 2 / np.maximum(
        np.abs(cp6["e_rhcp"].values) ** 2 + np.abs(cp6["e_lhcp"].values) ** 2, 1e-300
    )
    d6_co_db = 10 * np.log10(np.maximum(d6 * frac_r6, 1e-12))
    d6_x_db = 10 * np.log10(np.maximum(d6 * (1 - frac_r6), 1e-12))
    it6, ip6 = np.unravel_index(int(np.argmax(d6)), d6.shape)
    peak6_db = float(10 * np.log10(d6[it6, ip6]))

    ax = axes[0, 1]
    x6, y6 = assembled_cut(d6_co_db, angles6.theta, angles6.phi, 0.0)
    x6x, y6x = assembled_cut(d6_x_db, angles6.theta, angles6.phi, 0.0)
    ax.plot(x6, y6, label="AFM model, co-pol (RHCP)")
    ax.plot(x6x, y6x, "--", label="AFM model, cross-pol (LHCP)")
    overlay_measured_if_present(ax, "faenzi2019_fig6_measured.csv")
    ax.axhline(ref6["peak_copol_directivity"], color="0.4", lw=0.8, ls=":")
    ax.annotate(
        "measured 33 dBi", (-88, ref6["peak_copol_directivity"] + 0.9), fontsize=7
    )
    ax.set(
        xlim=(-90, 90),
        ylim=(-20, 40),
        xlabel="Off-axis angle [deg]",
        ylabel="Directivity [dBi]",
        title="(b) RHCP tilted 30$^\\circ$, 20 GHz, a = 10$\\lambda$",
    )
    ax.legend(loc="upper left", fontsize=7)

    xpd6 = float(d6_co_db[it6, ip6] - np.max(d6_x_db))
    val.check(
        "Fig6 beam direction",
        f"theta_peak = {np.rad2deg(angles6.theta[it6]):.1f} deg (design 30)",
        abs(np.rad2deg(angles6.theta[it6]) - ref6["beam_tilt"]) <= 1.0,
    )
    bound6_db = float(
        10 * np.log10(4 * np.pi * ant6.aperture.area / wavelength(f6) ** 2)
    )
    val.check(
        "Fig6 peak directivity",
        f"{peak6_db:.2f} dBi (paper: {ref6['peak_copol_directivity']} dBi measured "
        f"incl. losses; lossless bound {bound6_db:.1f} dBi)",
        ref6["peak_copol_directivity"] - 0.5 <= peak6_db <= bound6_db,
    )
    val.check(
        "Fig6 cross-pol discrimination",
        f"{xpd6:.1f} dB (paper: ~{ref6['crosspol_below_peak']} dB; "
        "first-order AFM underestimates)",
        xpd6 > 15.0,
    )

    # ------------------------------------------------------------------
    # Panel C - Fig. 10-style dispersion maps (RO3010 dual-band stack)
    # ------------------------------------------------------------------
    print("Panel C: leaky-wave dispersion maps (RO3010) ...")
    refd = load_reference("dualband")
    m_values = np.linspace(0.0, 0.45, 19)
    ax = axes[1, 0]
    ax2 = ax.twinx()
    for f_ghz, x_t in (
        (refd["f1"], refd["x_transparent_f1"]),
        (refd["f2"], refd["x_transparent_f2"]),
    ):
        f_hz = f_ghz * 1e9
        sw = solve_sw_transparent(
            x_t, f_hz, refd["substrate_eps_r"], refd["substrate_thickness"] * 1e-3
        )
        period = 2 * np.pi / sw.beta
        ds = dispersion_map(sw.x_op, m_values, period, f_hz)
        ax.plot(m_values, ds["beta_over_k0"].values, label=f"{f_ghz:.2f} GHz")
        ax2.plot(m_values, 1e3 * ds["alpha_over_k0"].values, "--")
    ax.set(
        xlabel="Modulation index $m$",
        ylabel="$\\beta_{LW}/k_0$ (solid)",
        title="(c) Leaky-wave dispersion, RO3010",
    )
    ax2.set_ylabel("$10^3 \\cdot \\alpha_{LW}/k_0$ (dashed)")
    ax2.grid(False)
    ax.legend(loc="upper left", fontsize=7)

    sw_check = solve_sw_transparent(
        refd["x_transparent_f1"],
        refd["f1"] * 1e9,
        refd["substrate_eps_r"],
        refd["substrate_thickness"] * 1e-3,
    )
    val.check(
        "Dual-band reactance pair f1",
        f"x_op = {sw_check.x_op / ETA_0:.3f} eta0 (paper: {refd['x_opaque_f1']} eta0)",
        abs(sw_check.x_op / ETA_0 - refd["x_opaque_f1"]) < 0.01,
    )

    # ------------------------------------------------------------------
    # Panel D - bandwidth: broadside directivity vs frequency (Fig. 5 antenna)
    # ------------------------------------------------------------------
    print("Panel D: directivity vs frequency (fixed modulation) ...")
    bg = ant5.bandwidth_gain(f5)
    freqs = np.linspace(28.75e9, 30.75e9, 9)
    cut_angles = AngleGrid.from_degrees(
        theta=np.linspace(0.0, 90.0, 181), phi=np.linspace(0.0, 355.0, 24)
    )
    d_broadside = []
    for f in freqs:
        # Quasi-static cladding: sheet reactance scales as f0/f.
        ant_f = ModulatedMetasurfaceAntenna(
            radius=ant5.radius,
            x_transparent=ant5.x_transparent * f5 / f,
            substrate=ro3003,
            modulation=ant5.modulation,
            n_rho=192,
            n_phi=96,
        )
        ff = ant_f.far_field(f, cut_angles)
        d = pattern_directivity(ff).values
        d_broadside.append(10 * np.log10(max(float(d[0, 0]), 1e-12)))

    ax = axes[1, 1]
    ax.plot(freqs / 1e9, d_broadside, "o-", label="AFM model (fixed modulation)")
    half_bw = 0.5 * bg.bandwidth * f5 / 1e9
    ax.axvspan(
        f5 / 1e9 - half_bw,
        f5 / 1e9 + half_bw,
        color="0.85",
        label=f"closed-form B = {100 * bg.bandwidth:.1f}%",
    )
    ax.axvspan(
        ref5["band_3db_low"],
        ref5["band_3db_high"],
        alpha=0.25,
        color="C1",
        label="measured 3 dB band",
    )
    ax.set(
        xlabel="Frequency [GHz]",
        ylabel="Broadside directivity [dBi]",
        title="(d) Bandwidth of the Fig. 5 design",
    )
    ax.legend(loc="lower center", fontsize=7)

    peak_d = max(d_broadside)
    in_band = [f for f, d in zip(freqs, d_broadside, strict=True) if d >= peak_d - 3.0]
    measured_bw = (ref5["band_3db_high"] - ref5["band_3db_low"]) / ref5["frequency"]
    val.check(
        "Fig5a bandwidth order",
        f"model 3dB band >= {1e-9 * (in_band[-1] - in_band[0]):.2f} GHz grid width; "
        f"closed-form B = {100 * bg.bandwidth:.1f}%, measured {100 * measured_bw:.1f}%",
        0.2 <= bg.bandwidth / measured_bw <= 5.0,
    )

    fig.suptitle(
        "Faenzi et al., Sci. Rep. 9:10178 (2019) - reduced-order AFM reproduction",
        y=1.0,
    )
    fig.tight_layout()
    paths = save_figure(fig, FIG_DIR / "faenzi2019_reproduction")
    print("\nSaved:", *[str(p) for p in paths])

    print(
        f"\nGain-bandwidth (Fig.5 design): GB_uniform = {bg.gb_uniform:.0f}, "
        f"GB_tapered = {bg.gb_tapered:.0f} (linear), v_g/c = {bg.v_g_over_c:.2f}"
    )
    val.finish()


if __name__ == "__main__":
    main()
