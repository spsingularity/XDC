#!/usr/bin/env python3
"""Generate the paper figures into paper/figs/ from the saved analysis results (*.npz):
  fig_spectrum.pdf -- selection-response spectrum across SED quantities (Table 1 / Sec 4.2)
  fig_band.pdf     -- mass-selection bias vs detection-band rest wavelength, two template
                      families, Balmer-break sign flip (Sec 4.3)

The .npz inputs are produced by:
  XDC_SAVE=results/spectrum.npz    code/run_jades_response_spectrum.py
  XDC_SAVE=results/band_qsf.npz    code/run_jades_band_response.py                 (FSPS QSF)
  XDC_SAVE=results/band_spline.npz XDC_TEMPL=.../tweak_spline.param code/run_jades_band_response.py
Set XDC_RESULTS to the directory holding them (default: ./results).
Usage: python3 code/make_figures.py
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.environ.get("XDC_RESULTS", os.path.join(os.path.dirname(__file__), "..", "results"))
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.frameon": False})


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"[missing {p} -- run the analysis with XDC_SAVE set]"); return None
    return np.load(p, allow_pickle=True)


# ===================== FIG 1: response spectrum =====================
S = load("spectrum.npz")
if S is not None:
    labels = [str(x) for x in S["labels"]]; slope = S["slope"]; se = S["se"]; units = [str(u) for u in S["units"]]
    order = np.argsort(slope)                       # most negative at bottom
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = ["C3" if slope[i] < 0 else "C0" for i in order]
    ax.barh(y, slope[order], xerr=se[order], color=colors, alpha=0.85, capsize=3, height=0.62)
    ax.axvline(0, color="k", lw=0.9)
    def pretty(s): return s.replace("dust A_V", r"dust $A_V$")
    lab = [pretty(labels[i]) + (r"  [mag]" if units[i] == "mag" else "") for i in order]
    ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.set_xlabel(r"bias per $1\sigma$ of detection-band flux   [dex, or mag for $A_V$]")
    ax.set_title(r"Selection-response spectrum (JADES $z\simeq7$–$9$)", fontsize=11)
    ax.text(0.97, 0.06, "blueward $A_V$, mass suppressed\nsSFR, youth inflated",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_spectrum.pdf")); plt.close(fig)
    print("wrote fig_spectrum.pdf")


# ===================== FIG 2: band response vs rest wavelength =====================
Q = load("band_qsf.npz"); P = load("band_spline.npz")
if Q is not None:
    bands = [str(b) for b in Q["bands"]]; rest = Q["rest"]; beta = Q["beta"]; se = Q["se"]
    drop = np.array([b in ("F090W", "F115W") for b in bands])   # Lyman-break dropout bands
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    # Balmer/4000A break shading
    ax.axvspan(3646, 4000, color="0.85", zorder=0, label="Balmer / 4000 Å break")
    ax.axhline(0, color="k", lw=0.9)
    m = ~drop
    ax.errorbar(rest[m], beta[m], yerr=se[m], marker="o", ls="-", color="C0", capsize=3,
                label="FSPS QSF templates", zorder=5)
    ax.errorbar(rest[drop], beta[drop], yerr=se[drop], marker="o", ls="none", color="C0",
                mfc="white", capsize=3, alpha=0.6, zorder=4, label="Lyman-break dropout bands")
    if P is not None:
        bs = [str(b) for b in P["bands"]]; rs = P["rest"]; bt = P["beta"]; ses = P["se"]
        dp = np.array([b in ("F090W", "F115W") for b in bs]); mm = ~dp
        ax.errorbar(rs[mm], bt[mm], yerr=ses[mm], marker="s", ls="--", color="C1", mfc="white",
                    capsize=3, label="spline templates (independent)", zorder=5)
    # annotate a few key bands
    for b, r, v in zip(bands, rest, beta):
        if b in ("F150W", "F277W", "F444W", "F410M"):
            ax.annotate(b, (r, v), textcoords="offset points", xytext=(4, 5), fontsize=8)
    ax.set_xlabel("rest wavelength of detection band at $z\\simeq8$   [Å]")
    ax.set_ylabel(r"mass bias  $\mathrm{d}\log_{10}M_\star/\mathrm{d}\sigma_{\rm band}$   [dex]")
    ax.set_title("Sign of the mass-selection bias vs selection band", fontsize=11)
    ax.text(1950, -0.145, "UV bands:\nbuffer the SMF", fontsize=9, color="C0", ha="center")
    ax.text(4050, 0.30, "optical bands:\ninflate the SMF", fontsize=9, color="C1", ha="center")
    ax.legend(loc="upper left", fontsize=8.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_band.pdf")); plt.close(fig)
    print("wrote fig_band.pdf")
