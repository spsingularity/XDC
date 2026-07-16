#!/usr/bin/env python3
"""
XDC on REAL stellar masses: the two-depth matched-overlap test applied to the SED-inferred
STELLAR MASS of JADES DR5 GOODS-S z~7-9 galaxies -- the measurement the UV-flux runners
(run_jades_crossdepth.py) could only make on flux, and which the paper's mass-function claim
actually requires.

WHY THIS IS THE REAL TEST (and why rho is PHYSICAL here, not injected):
  The correlated selection bias is corr(selection noise, mass-inference noise) = rho. For
  stellar mass the selection is on the (noisy) detection-band flux, while the outcome is the
  full-SED stellar mass -- and the SAME medium-depth photometric noise that scatters a galaxy
  above the detection threshold ALSO enters its SED fit and inflates the inferred mass. So the
  selection<->mass noise correlation is built into the photometry; we do not assume rho, we
  measure the bias it produces.

METHOD (controlled degradation, matched-overlap on mass):
  1. Take the DEEP tier (F150W_WHT above the field median): high-S/N galaxies whose eazy
     stellar mass M_deep is the low-noise reference.
  2. Fit M_deep with eazy at FIXED photo-z (z_peak), FSPS QSF templates.
  3. Degrade EVERY band from its deep noise to the MEDIUM-tier median noise, using the real
     per-band WHT ratio (one shared noise draw per galaxy, so the detection-band up-scatter and
     the mass up-scatter are the SAME event -> physical rho).
  4. Refit the degraded photometry -> M_medium; apply the medium selection (F150W S/N > 5).
  5. SIGNAL = <logM_medium - logM_deep> among medium-survivors (selection conditions on the
     medium noise -> correlated-selection bias). NULL = the same shift with survival decided by
     the DEEP-band S/N (uncorrelated with the medium noise draw) -> isolates any pure
     degradation bias, which we subtract. The net, proximity-dependent shift is the bias.

Masses use the cheap, exact-for-differences path: at FIXED z the eazy stellar mass is
  M = (coeffs . template_mass) x C(z), and the per-object C(z) cancels in logM_med - logM_deep
(validated against eazy sps_parameters to <0.01 dex; see validate_cheapmass).

Usage: python3 code/run_jades_mass_crossdepth.py [jades_catalog.fits]
Requires: eazy (+ its eazy-photoz data), astropy, numpy, scipy.
"""
import os, sys, time, numpy as np
from astropy.io import fits
from astropy.table import Table

# --- eazy setup ---
import eazy, eazy.photoz
EZDAT = os.path.join(os.path.dirname(eazy.__file__), "data", "eazy-photoz")
# template set is overridable (XDC_TEMPL) for robustness checks across template families
TEMPL = os.environ.get("XDC_TEMPL",
                       os.path.join(EZDAT, "templates", "fsps_full", "tweak_fsps_QSF_12_v3.param"))
FRES  = os.path.join(EZDAT, "filters", "FILTER.RES.latest")
TMASS = np.asarray(Table.read(TEMPL + ".fits")["mass"])   # per-template stellar mass

# JADES band -> FILTER.RES.latest number (NIRCam wides+mediums verified; ACS/WFC t81)
FILT = {"F090W":363,"F115W":364,"F150W":365,"F200W":366,"F277W":375,"F335M":381,
        "F356W":376,"F410M":383,"F444W":377,"F435W":233,"F606W":236,"F775W":238,"F814W":239}
SEL_BAND = "F150W"        # detection/selection band (rest-UV at z~7-9)
SNCUT    = 5.0
R_REAL   = 24             # degradation-noise realisations
WORK = os.environ.get("XDC_WORK", "/tmp/xdc_mass"); os.makedirs(WORK, exist_ok=True)
rng = np.random.default_rng(7)


def load_deep_tier(catfile):
    """z~7-9 sample; return deep-tier galaxies with per-band flux, err, and medium-target err."""
    h = fits.open(catfile, memmap=True)
    CIRC, FLAG, PZ = h[5].data, h[2].data, h[11].data
    def c(hdu, name): return np.asarray(hdu[name], float)
    zpk, pg7 = c(PZ, "z_peak"), c(PZ, "Prob_gt_7")
    fsel, esel = c(CIRC, f"{SEL_BAND}_CIRC1"), c(CIRC, f"{SEL_BAND}_CIRC1_e")
    wsel = c(FLAG, f"{SEL_BAND}_WHT")
    base = (np.isfinite(zpk) & (zpk >= 7) & (zpk < 9) & (pg7 > 0.7)
            & np.isfinite(fsel) & (fsel > 0) & np.isfinite(esel) & (esel > 0)
            & np.isfinite(wsel) & (wsel > 0) & (fsel/esel > SNCUT))
    # deep vs medium split by selection-band weight (same field)
    wmed = np.median(wsel[base])
    deep = base & (wsel > wmed)
    idx = np.where(deep)[0]
    flux = {b: c(CIRC, f"{b}_CIRC1")[idx] for b in FILT}
    ferr = {b: c(CIRC, f"{b}_CIRC1_e")[idx] for b in FILT}
    wht  = {b: c(FLAG, f"{b}_WHT")[idx] for b in FILT}
    # medium-target noise per band: scale deep err by sqrt(WHT_deep / WHT_medium_median)
    medmask = base & (wsel <= wmed)
    etarget = {}
    for b in FILT:
        wmed_b = np.median(c(FLAG, f"{b}_WHT")[medmask & np.isfinite(c(FLAG, f"{b}_WHT"))])
        r = np.sqrt(np.clip(wht[b]/max(wmed_b, 1e-30), 1.0, None))   # deep->medium (>=1)
        etarget[b] = ferr[b]*r
    z = zpk[idx]
    contrast = np.sqrt(np.median(wsel[deep])/np.median(wsel[medmask]))
    print(f"  z~7-9 deep tier: N={idx.size}; deep->medium {SEL_BAND} noise contrast ~{contrast:.2f}x")
    return z, flux, ferr, etarget


def write_cat(path, z, flux, ferr):
    hdr = "# id z_spec " + " ".join(f"F{FILT[b]} E{FILT[b]}" for b in FILT)
    lines = [hdr]
    for j in range(len(z)):
        r = [str(j+1), f"{z[j]:.4f}"]
        for b in FILT:
            fv, ev = flux[b][j], ferr[b][j]
            if not (np.isfinite(fv) and np.isfinite(ev) and ev > 0): fv, ev = -99., -99.
            r += [f"{fv:.4e}", f"{ev:.4e}"]
        lines.append(" ".join(r))
    open(path, "w").write("\n".join(lines) + "\n")


def make_photoz(catfile, tempfilt=None):
    trans = os.path.join(WORK, "xdc.translate"); open(trans, "w").write("")
    params = dict(CATALOG_FILE=catfile, MAIN_OUTPUT_FILE=os.path.join(WORK, "xdc"),
                  FILTERS_RES=FRES, TEMPLATES_FILE=TEMPL, FIX_ZSPEC="y", PRIOR_ABZP=31.4,
                  MW_EBV=0.0, CAT_HAS_EXTCORR="n", Z_MIN=6.5, Z_MAX=9.5, Z_STEP=0.02,
                  N_MIN_COLORS=3, SYS_ERR=0.05)
    ez = eazy.photoz.PhotoZ(param_file=None, params=params, translate_file=trans,
                            load_prior=False, n_proc=0, tempfilt=tempfilt)
    return ez


def logmass(ez):
    """cheap relative log10 M* (per-object distance const cancels in differences at fixed z)."""
    C = ez.coeffs_best                          # NOBJ x NTEMP
    return np.log10((C * TMASS[None, :]).sum(axis=1))


def main():
    t0 = time.time()
    catfile = sys.argv[1] if len(sys.argv) > 1 else "data/jades_dr5_gds_phot.fits"
    z, flux, ferr, etarget = load_deep_tier(catfile)

    # --- deep reference fit ---
    deepcat = os.path.join(WORK, "deep.cat"); write_cat(deepcat, z, flux, ferr)
    ez = make_photoz(deepcat)
    ez.fit_catalog(n_proc=0, verbose=False)
    lm_deep = logmass(ez)
    tempfilt = ez.tempfilt                       # reuse grid across realisations
    print(f"  deep fit done ({time.time()-t0:.0f}s); log10 M* median={np.nanmedian(lm_deep):.2f}")

    # --- degradation realisations ---
    # SIGNAL: survival decided by the medium-depth F150W noise draw that ALSO enters the mass fit
    #   -> selection conditions on the mass-affecting up-scatter (correlated selection).
    # NULL: survival decided by an INDEPENDENT medium-depth F150W draw (same noise level, NOT in
    #   the mass fit) -> selects the same brightness distribution but does not condition on the
    #   mass noise, so it carries only the brightness-dependent degradation bias. SIGNAL - NULL
    #   isolates the correlated-selection term (the mass analogue of the flux test's redder-band
    #   null in run_jades_crossdepth.py).
    add_sel = np.sqrt(np.clip(etarget[SEL_BAND]**2 - ferr[SEL_BAND]**2, 0, None))
    sig_shift = []; null_shift = []; prox = []
    for r in range(R_REAL):
        fdeg, edeg = {}, {}
        for b in FILT:
            add = np.sqrt(np.clip(etarget[b]**2 - ferr[b]**2, 0, None))
            fdeg[b] = flux[b] + rng.standard_normal(len(z))*add
            edeg[b] = etarget[b]
        rc = os.path.join(WORK, f"deg_{r}.cat"); write_cat(rc, z, fdeg, edeg)
        ezr = make_photoz(rc, tempfilt=tempfilt)
        ezr.fit_catalog(n_proc=0, verbose=False)
        dlm = logmass(ezr) - lm_deep
        sn_A = fdeg[SEL_BAND]/edeg[SEL_BAND]                              # shared with mass fit
        fsel_B = flux[SEL_BAND] + rng.standard_normal(len(z))*add_sel    # independent draw
        sn_B = fsel_B/etarget[SEL_BAND]                                  # not in mass fit
        surv_A = sn_A > SNCUT; surv_B = sn_B > SNCUT
        good = np.isfinite(dlm)
        sig_shift.append(np.nanmean(dlm[surv_A & good]))
        null_shift.append(np.nanmean(dlm[surv_B & good]))
        prox.append((sn_A, sn_B, dlm, good))
        print(f"    real {r+1:2d}/{R_REAL}: signal={sig_shift[-1]:+.4f}  null={null_shift[-1]:+.4f}  "
              f"net={sig_shift[-1]-null_shift[-1]:+.4f}  Nsurv={int((surv_A&good).sum())}", flush=True)

    sig = np.array(sig_shift); nul = np.array(null_shift); net = sig - nul
    se = net.std()/np.sqrt(len(net))
    print("\n" + "="*74)
    print("REAL two-depth STELLAR-MASS matched-overlap (JADES DR5 GOODS-S, z~7-9):")
    print(f"  SIGNAL (medium-selected, correlated)   <dlogM> = {sig.mean():+.4f} +/- {sig.std():.4f} dex")
    print(f"  NULL   (independent-draw, brightness-matched) = {nul.mean():+.4f} +/- {nul.std():.4f} dex")
    print(f"  NET correlated-selection mass bias = {net.mean():+.4f} +/- {se:.4f} dex "
          f"({net.mean()/(se+1e-9):.1f} sigma)")
    # proximity: net (signal - null) vs medium S/N -- selection effect should rise toward the limit
    print("\n  net (signal - null) mass boost vs proximity to the medium limit:")
    snA = np.concatenate([p[0][p[3]] for p in prox]); dA = np.concatenate([p[2][p[3]] for p in prox])
    snB = np.concatenate([p[1][p[3]] for p in prox]); dB = np.concatenate([p[2][p[3]] for p in prox])
    bins = [(5,6),(6,8),(8,12),(12,1e9)]; xc = []; ynet = []; ysig = []; ynul = []
    for lo, hi in bins:
        mA = (snA >= lo) & (snA < hi); mB = (snB >= lo) & (snB < hi)
        if mA.sum() < 20 or mB.sum() < 20: continue
        xc.append(0.5*(lo + (hi if hi < 1e9 else 16)))
        ysig.append(dA[mA].mean()); ynul.append(dB[mB].mean()); ynet.append(dA[mA].mean()-dB[mB].mean())
        print(f"    medium S/N {lo:>2.0f}-{hi if hi<1e9 else 999:>3.0f}: "
              f"signal={ysig[-1]:+.4f}  null={ynul[-1]:+.4f}  net={ynet[-1]:+.4f} dex (N={mA.sum()})")
    save = os.environ.get("XDC_SAVE")
    if save:
        np.savez(save, net=net.mean(), net_se=se, sig=sig.mean(), nul=nul.mean(),
                 xc=np.array(xc), ynet=np.array(ynet), ysig=np.array(ysig), ynul=np.array(ynul))
        print(f"  saved -> {save}")
    print(f"\n  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
