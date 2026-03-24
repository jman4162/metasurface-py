"""metasurface-py Interactive Demo.

A Streamlit web app for designing and analyzing programmable
electromagnetic metasurfaces. Three tabs: Beam Pattern Designer,
RIS Link Calculator, and Optimization.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from metasurface_py.channels import RISLink, free_space_path_loss_db
from metasurface_py.core.types import AngleGrid, Position3D
from metasurface_py.elements import DiscretePhaseSpace, PhaseOnlyCell
from metasurface_py.elements.states import ContinuousPhaseSpace
from metasurface_py.em import (
    far_field_pattern,
    half_power_beamwidth,
    peak_gain_db,
    sidelobe_level,
    steering_phase,
)
from metasurface_py.geometry import RectangularLattice
from metasurface_py.optimize import MaxGainObjective, relax_then_quantize
from metasurface_py.plotting import (
    plot_pattern_2d,
    plot_pattern_comparison,
    plot_convergence,
    plot_state_map,
    set_publication_style,
)
from metasurface_py.surfaces import Metasurface

st.set_page_config(
    page_title="metasurface-py",
    page_icon="📡",
    layout="wide",
)

st.title("📡 metasurface-py Interactive Demo")
st.caption("Design, analyze, and optimize programmable electromagnetic metasurfaces")

set_publication_style()

tab1, tab2, tab3 = st.tabs([
    "🎯 Beam Pattern Designer",
    "📶 RIS Link Calculator",
    "⚡ Optimization",
])

# ── Tab 1: Beam Pattern Designer ──────────────────────────────

with tab1:
    col_params, col_plots = st.columns([1, 2])

    with col_params:
        st.subheader("Parameters")
        nx = st.slider("Array size (N×N)", 4, 48, 16, 4, key="t1_nx")
        freq_ghz = st.slider("Frequency [GHz]", 1.0, 60.0, 28.0, 0.5, key="t1_freq")
        theta_deg = st.slider("Steering angle θ [deg]", 0, 80, 30, 1, key="t1_theta")
        num_bits = st.selectbox("Quantization", [None, 1, 2, 3], index=2,
                                format_func=lambda x: "Continuous" if x is None else f"{x}-bit",
                                key="t1_bits")

    freq = freq_ghz * 1e9
    lam = 3e8 / freq
    lattice = RectangularLattice(nx=nx, ny=nx, dx=lam/2, dy=lam/2)

    if num_bits is not None:
        cell = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=num_bits))
    else:
        cell = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
    surface = Metasurface(lattice=lattice, cell=cell)

    phase = steering_phase(lattice, np.radians(theta_deg), 0.0, freq)
    state = surface.set_state(phase)
    if num_bits is not None:
        state = state.quantize()

    angles = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.array([0.0]),
    )
    angles_full = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 180),
        phi=np.linspace(0, 2 * np.pi - 0.01, 72),
    )

    pattern = far_field_pattern(surface, state, freq, angles)
    pattern_full = far_field_pattern(surface, state, freq, angles_full)
    gain = peak_gain_db(pattern_full)
    hpbw = np.rad2deg(half_power_beamwidth(pattern, cut_phi=0.0))

    with col_plots:
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak Gain", f"{gain:.1f} dBi")
        m2.metric("HPBW", f"{hpbw:.1f}°")
        m3.metric("Elements", f"{lattice.num_elements}")

        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plot_pattern_2d(pattern, cut_phi=0.0, ax=ax1)
        ax1.set_xlim([0, 90])
        ax1.set_ylim([-30, 0])
        ax1.set_title("Far-Field Pattern (φ = 0° cut)")

        plot_state_map(state, nx=nx, ny=nx, ax=ax2)
        ax2.set_title("Phase Map")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── Tab 2: RIS Link Calculator ────────────────────────────────

with tab2:
    col_params2, col_plots2 = st.columns([1, 2])

    with col_params2:
        st.subheader("Link Parameters")
        n_ris = st.slider("RIS size (N×N)", 4, 48, 16, 4, key="t2_nx")
        freq2_ghz = st.slider("Frequency [GHz]", 1.0, 60.0, 28.0, 0.5, key="t2_freq")
        tx_z = st.slider("TX height [m]", 5.0, 100.0, 20.0, 5.0, key="t2_txz")
        rx_x = st.slider("RX distance [m]", 5.0, 100.0, 10.0, 5.0, key="t2_rxx")
        tx_power = st.slider("TX power [dBm]", 0, 40, 20, 1, key="t2_txp")
        bits_ris = st.selectbox("RIS quantization", [None, 1, 2, 3], index=2,
                                format_func=lambda x: "Continuous" if x is None else f"{x}-bit",
                                key="t2_bits")

    freq2 = freq2_ghz * 1e9
    lam2 = 3e8 / freq2
    tx2 = Position3D(x=0.0, y=0.0, z=tx_z)
    rx2 = Position3D(x=rx_x, y=0.0, z=1.5)
    noise_dbm = -90.0

    lattice2 = RectangularLattice(nx=n_ris, ny=n_ris, dx=lam2/2, dy=lam2/2)
    cell_cont2 = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
    surface_cont2 = Metasurface(lattice=lattice2, cell=cell_cont2)
    link_cont = RISLink(surface=surface_cont2, tx=tx2, rx=rx2, freq=freq2, include_direct=True)

    opt_state = link_cont.optimal_state_continuous()
    snr_opt = link_cont.snr_db(opt_state, float(tx_power), noise_dbm)

    if bits_ris is not None:
        cell_q = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=bits_ris))
        state_q = opt_state.quantize(cell_q.state_space.codebook)
        surface_q = Metasurface(lattice=lattice2, cell=cell_q)
        link_q = RISLink(surface=surface_q, tx=tx2, rx=rx2, freq=freq2, include_direct=True)
        snr_q = link_q.snr_db(state_q, float(tx_power), noise_dbm)
    else:
        snr_q = snr_opt

    d_direct = tx2.distance_to(rx2)
    pl_direct = free_space_path_loss_db(d_direct, freq2)
    snr_no_ris = float(tx_power) - pl_direct - noise_dbm

    with col_plots2:
        m1, m2, m3 = st.columns(3)
        m1.metric("SNR (Optimal)", f"{snr_opt:.1f} dB")
        label = "Continuous" if bits_ris is None else f"{bits_ris}-bit"
        m2.metric(f"SNR ({label})", f"{snr_q:.1f} dB")
        m3.metric("SNR (No RIS)", f"{snr_no_ris:.1f} dB")

        # SNR vs array size
        sizes = [4, 8, 12, 16, 20, 24, 32]
        snrs_sweep = []
        for n in sizes:
            lat_s = RectangularLattice(nx=n, ny=n, dx=lam2/2, dy=lam2/2)
            cell_s = PhaseOnlyCell(state_space=ContinuousPhaseSpace())
            surf_s = Metasurface(lattice=lat_s, cell=cell_s)
            link_s = RISLink(surface=surf_s, tx=tx2, rx=rx2, freq=freq2, include_direct=True)
            os = link_s.optimal_state_continuous()
            snrs_sweep.append(link_s.snr_db(os, float(tx_power), noise_dbm))

        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.plot([n**2 for n in sizes], snrs_sweep, "o-", label="Optimal RIS")
        ax.axhline(y=snr_no_ris, color="gray", linestyle=":", label="No RIS")
        ax.set_xlabel("Number of RIS Elements")
        ax.set_ylabel("SNR [dB]")
        ax.set_title("SNR vs RIS Array Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)


# ── Tab 3: Optimization ───────────────────────────────────────

with tab3:
    col_params3, col_plots3 = st.columns([1, 2])

    with col_params3:
        st.subheader("Optimization Settings")
        n_opt = st.slider("Array size (N×N)", 4, 24, 10, 2, key="t3_nx")
        freq3_ghz = st.slider("Frequency [GHz]", 1.0, 60.0, 10.0, 0.5, key="t3_freq")
        theta_opt = st.slider("Target θ [deg]", 0, 80, 30, 5, key="t3_theta")
        bits_opt = st.selectbox("Quantization", [1, 2, 3], index=1, key="t3_bits",
                                format_func=lambda x: f"{x}-bit")
        maxiter = st.slider("Max iterations", 10, 200, 50, 10, key="t3_maxiter")
        run_btn = st.button("🚀 Run Optimization", key="t3_run")

    freq3 = freq3_ghz * 1e9
    lam3 = 3e8 / freq3
    lattice3 = RectangularLattice(nx=n_opt, ny=n_opt, dx=lam3/2, dy=lam3/2)
    cell3 = PhaseOnlyCell(state_space=DiscretePhaseSpace(num_bits=bits_opt))
    surface3 = Metasurface(lattice=lattice3, cell=cell3)

    angles_opt = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 90),
        phi=np.linspace(0, 2 * np.pi - 0.01, 36),
    )
    angles_plot3 = AngleGrid(
        theta=np.linspace(0.01, np.pi - 0.01, 360),
        phi=np.array([0.0]),
    )

    if run_btn:
        target = np.radians(theta_opt)
        obj = MaxGainObjective(target, 0.0, angles_opt)

        # Baseline
        phase_bl = steering_phase(lattice3, target, 0.0, freq3)
        state_bl = surface3.set_state(phase_bl).quantize()
        pattern_bl = far_field_pattern(surface3, state_bl, freq3, angles_plot3)

        # Optimize
        with st.spinner("Optimizing..."):
            result = relax_then_quantize(
                obj, surface3, freq3, angles_opt,
                continuous_method="L-BFGS-B",
                refine=True, maxiter=maxiter, seed=42,
            )

        pattern_opt = far_field_pattern(surface3, result.state, freq3, angles_plot3)

        with col_plots3:
            st.success(f"Done in {result.runtime_seconds:.1f}s")

            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            plot_pattern_comparison(
                [(pattern_bl, "Analytical+Quantize"),
                 (pattern_opt, "Optimized+Quantize")],
                cut_phi=0.0, ax=ax1,
            )
            ax1.set_xlim([0, 90])
            ax1.set_ylim([-30, 0])
            ax1.set_title("Pattern Comparison")

            plot_convergence(
                {"Objective": result.convergence_history},
                ax=ax2, ylabel="Neg. Gain [dBi]",
            )
            ax2.set_title("Convergence")
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
    else:
        with col_plots3:
            st.info("Click 'Run Optimization' to start.")
