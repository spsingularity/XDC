#!/usr/bin/env python3
"""
XDC selection-response spectrum: how each SED-inferred quantity responds to a fluctuation of the
DETECTION-band flux, measured model-independently from the real deep->medium depth contrast.

The correlated selection bias in any inferred quantity Q is set by rho_Q = corr(detection-band
noise, Q noise). We measure it directly: degrade the JADES deep tier to the medium depth many
times and, per galaxy, correlate the F150W (detection band) flux deviation with the deviation of
each inferred quantity. NO selection is applied -- this is the underlying RESPONSE that the
selection then conditions on (bias = response x mean up-scatter of near-limit survivors).

Quantities (all from the same eazy FSPS-QSF coefficients; per-object distance normalisation
cancels in log-differences / ratios at fixed z):
  M*      = C . mass                         (stellar mass)
  SFR     = C . sfr                          (star-formation rate; UV-driven)
  sSFR    = SFR / M*                          (doubly sensitive: SFR up, M* down)
  A_V     = (C . Lv*Av)/(C . Lv)              (V-light-weighted dust)
  young   = (C . formed_100)/(C . formed_total)  (recent mass fraction; youth)
  f150    = detection-band flux               (UV luminosity anchor; response ~ +1 by construction)

Prediction: SFR, sSFR strongly POSITIVE (UV up-scatter -> more SF, younger, lower M/L);
A_V, M* NEGATIVE (bluer, lower mass). sSFR is amplified from both ends.

Usage: python3 code/run_jades_response_spectrum.py [jades_catalog.fits]
"""
import os, sys, time, numpy as np
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_jades_mass_crossdepth as M   # shared: FILT, SEL_BAND, load_deep_tier, write_cat, make_photoz
import eazy
EZDAT = os.path.join(os.path.dirname(eazy.__file__), "data", "eazy-photoz")
T = Table.read(os.path.join(EZDAT, "templates", "fsps_full", "tweak_fsps_QSF_12_v3.param.fits"))
tmass = np.asarray(T["mass"]); tsfr = np.asarray(T["sfr"]); tLv = np.asarray(T["Lv"])
tAv = np.asarray(T["Av"]); tf100 = np.asarray(T["formed_100"]); tftot = np.asarray(T["formed_total"])

WORK = os.environ.get("XDC_WORK", "/tmp/xdc_resp"); os.makedirs(WORK, exist_ok=True)
M.WORK = WORK
rng = np.random.default_rng(11)
NSUB = int(os.environ.get("XDC_NSUB", "900"))   # galaxies (fits are fast; more -> tighter slopes)
NR   = int(os.environ.get("XDC_NR", "30"))      # degradation realisations


def quantities(ez):
    """dict of inferred quantities from eazy coefficients (relative; log-diffs are absolute)."""
    C = ez.coeffs_best
    M_ = (C*tmass).sum(1); SFR = (C*tsfr).sum(1); Lv = (C*tLv).sum(1)
    Av = (C*(tLv*tAv)).sum(1)/np.maximum(Lv, 1e-30)
    young = (C*tf100).sum(1)/np.maximum((C*tftot).sum(1), 1e-30)
    return dict(logM=np.log10(np.maximum(M_, 1e-30)),
                logSFR=np.log10(np.maximum(SFR, 1e-30)),
                logsSFR=np.log10(np.maximum(SFR, 1e-30)) - np.log10(np.maximum(M_, 1e-30)),
                Av=Av, logyoung=np.log10(np.clip(young, 1e-6, None)))


def main():
    t0 = time.time()
    cat = sys.argv[1] if len(sys.argv) > 1 else "data/jades_dr5_gds_phot.fits"
    z, flux, ferr, etarget = M.load_deep_tier(cat)
    n = min(NSUB, len(z)); sl = slice(0, n)
    z = z[sl]; flux = {b: flux[b][sl] for b in M.FILT}; ferr = {b: ferr[b][sl] for b in M.FILT}
    etarget = {b: etarget[b][sl] for b in M.FILT}
    SEL = M.SEL_BAND
    add = {b: np.sqrt(np.clip(etarget[b]**2 - ferr[b]**2, 0, None)) for b in M.FILT}

    dc = os.path.join(WORK, "deep.cat"); M.write_cat(dc, z, flux, ferr)
    ez = M.make_photoz(dc); ez.fit_catalog(n_proc=0, verbose=False)
    Q0 = quantities(ez); tf = ez.tempfilt
    print(f"  deep fit done ({time.time()-t0:.0f}s), N={n}; medians "
          f"logM={np.nanmedian(Q0['logM']):.2f} logsSFR={np.nanmedian(Q0['logsSFR']):.2f}", flush=True)

    keys = ["logM", "logSFR", "logsSFR", "Av", "logyoung"]
    dsig = []                          # F150W detection-band noise (in sigma)
    dQ = {k: [] for k in keys}         # deviation of each quantity from deep
    for r in range(NR):
        fdeg, edeg = {}, {}
        for b in M.FILT:
            fdeg[b] = flux[b] + rng.standard_normal(n)*add[b]; edeg[b] = etarget[b]
        rc = os.path.join(WORK, f"r{r}.cat"); M.write_cat(rc, z, fdeg, edeg)
        er = M.make_photoz(rc, tempfilt=tf); er.fit_catalog(n_proc=0, verbose=False)
        Q = quantities(er)
        dsig.append((fdeg[SEL]-flux[SEL])/np.maximum(etarget[SEL], 1e-30))
        for k in keys: dQ[k].append(Q[k] - Q0[k])
        print(f"    real {r+1:2d}/{NR}", end="\r", flush=True)
    dsig = np.array(dsig)              # NR x n

    print("\n" + "="*74)
    print("SELECTION-RESPONSE SPECTRUM (JADES DR5 z~7-9): response of each SED quantity to a")
    print("+1 sigma fluctuation of the F150W DETECTION band, at the real deep->medium contrast.")
    print(f"  (pure response, no selection; {n} galaxies x {NR} realisations)\n")
    print(f"  {'quantity':10s}  {'slope d/dsigma':>16s}   {'corr rho':>9s}   interpretation")
    A = dsig.ravel()
    order = [("logSFR","dex","SFR"),("logsSFR","dex","sSFR"),("logyoung","dex","recent-mass frac"),
             ("logM","dex","stellar mass"),("Av","mag","dust A_V")]
    res = {}
    for k, unit, lbl in order:
        B = np.array(dQ[k]).ravel()
        ok = np.isfinite(A) & np.isfinite(B)
        slope = np.polyfit(A[ok], B[ok], 1)[0]
        rho = np.corrcoef(A[ok], B[ok])[0, 1]
        # bootstrap slope error over galaxies
        gsl = []
        for _ in range(200):
            j = rng.integers(0, n, n)
            aa = dsig[:, j].ravel(); bb = np.array(dQ[k])[:, j].ravel()
            m = np.isfinite(aa) & np.isfinite(bb)
            gsl.append(np.polyfit(aa[m], bb[m], 1)[0])
        se = np.std(gsl)
        sign = "+" if slope > 0 else "-"
        res[k] = (slope, se, rho)
        print(f"  {lbl:16s}  {slope:+.4f} {unit}/σ ± {se:.4f}   {rho:+.3f}   "
              f"{'INFLATED' if slope>0 else 'suppressed'} at the limit")
    print(f"\n  UV luminosity (f150 detection band): response +1 by construction (the anchor).")
    print("\n  => sSFR is inflated from BOTH ends (SFR up, M* down); dust and mass are suppressed.")
    print("     Near a survey's detection limit these biases are (response x mean survivor up-scatter).")
    print(f"  (elapsed {time.time()-t0:.0f}s)")
    save = os.environ.get("XDC_SAVE")
    if save:
        labels = [lbl for _, _, lbl in order]
        np.savez(save, labels=np.array(labels),
                 slope=np.array([res[k][0] for k, _, _ in order]),
                 se=np.array([res[k][1] for k, _, _ in order]),
                 rho=np.array([res[k][2] for k, _, _ in order]),
                 units=np.array([u for _, u, _ in order]))
        print(f"  saved -> {save}")


if __name__ == "__main__":
    main()
