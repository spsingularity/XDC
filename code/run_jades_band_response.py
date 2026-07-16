"""
XDC band-response of the stellar-mass selection bias: how the correlated-selection MASS bias
depends on the rest-frame wavelength of the DETECTION band.

The sign of rho_mass = corr(detection-band noise, inferred-mass noise) should depend on what the
detection band constrains: a rest-UV band traces young stars (up-scatter -> younger/lower M/L ->
LOWER mass, rho<0, SMF buffered), while a rest-optical band past the Balmer break traces the mass
(up-scatter -> HIGHER mass, rho>0, SMF inflated). We measure it directly: degrade all bands of the
JADES deep tier to medium depth many times and regress the inferred-mass deviation on EACH band's
independent noise (a multivariate partial response). No selection is applied.

Result: the mass-response slope d(logM)/d(band sigma) vs the band's rest wavelength at z~8, crossing
zero near the 4000A/Balmer break -> the sign of the SMF correlated-selection bias is set by the
survey's selection band. UV-dropout selection (JADES) buffers the SMF; rest-optical selection
inflates it.

Usage: python3 code/run_jades_band_response.py [jades_catalog.fits]
"""
import os, sys, time, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_jades_mass_crossdepth as M

WORK = os.environ.get("XDC_WORK", "/tmp/xdc_band"); os.makedirs(WORK, exist_ok=True)
M.WORK = WORK
rng = np.random.default_rng(19)
NSUB = int(os.environ.get("XDC_NSUB", "900"))
NR   = int(os.environ.get("XDC_NR", "40"))

# NIRCam pivot wavelengths (micron) for the fitted bands -> rest frame divides by (1+z)
PIVOT = {"F090W":0.901,"F115W":1.154,"F150W":1.501,"F200W":1.990,"F277W":2.786,
         "F335M":3.365,"F356W":3.563,"F410M":4.092,"F444W":4.421}
NIRCAM = [b for b in ["F090W","F115W","F150W","F200W","F277W","F335M","F356W","F410M","F444W"]
          if b in M.FILT]


def main():
    t0 = time.time()
    cat = sys.argv[1] if len(sys.argv) > 1 else "data/jades_dr5_gds_phot.fits"
    z, flux, ferr, etarget = M.load_deep_tier(cat)
    n = min(NSUB, len(z)); sl = slice(0, n)
    z = z[sl]; flux = {b: flux[b][sl] for b in M.FILT}; ferr = {b: ferr[b][sl] for b in M.FILT}
    etarget = {b: etarget[b][sl] for b in M.FILT}
    zmed = float(np.median(z))
    add = {b: np.nan_to_num(np.sqrt(np.clip(etarget[b]**2 - ferr[b]**2, 0, None)), nan=0.0)
           for b in M.FILT}
    esafe = {b: np.where(np.isfinite(etarget[b]) & (etarget[b] > 0), etarget[b], np.inf)
             for b in M.FILT}   # missing band -> zero noise leverage in the regression

    dc = os.path.join(WORK, "deep.cat"); M.write_cat(dc, z, flux, ferr)
    ez = M.make_photoz(dc); ez.fit_catalog(n_proc=0, verbose=False)
    lm0 = M.logmass(ez); tf = ez.tempfilt
    print(f"  deep fit done ({time.time()-t0:.0f}s), N={n}, z_med={zmed:.2f}", flush=True)

    # collect per-(realisation, galaxy) band noise (sigma) and logM deviation
    Xrows = []; yrows = []
    for r in range(NR):
        fdeg = {}; edeg = {}
        for b in M.FILT:
            fdeg[b] = flux[b] + rng.standard_normal(n)*add[b]; edeg[b] = etarget[b]
        rc = os.path.join(WORK, f"r{r}.cat"); M.write_cat(rc, z, fdeg, edeg)
        er = M.make_photoz(rc, tempfilt=tf); er.fit_catalog(n_proc=0, verbose=False)
        dlm = M.logmass(er) - lm0
        band_sig = np.array([np.nan_to_num((fdeg[b]-flux[b])/esafe[b], nan=0.0,
                             posinf=0.0, neginf=0.0) for b in NIRCAM]).T  # n x nb, finite
        ok = np.isfinite(dlm) & np.isfinite(band_sig).all(1)
        Xrows.append(band_sig[ok]); yrows.append(dlm[ok])
        print(f"    real {r+1:2d}/{NR}", end="\r", flush=True)
    X = np.vstack(Xrows); y = np.concatenate(yrows)          # (NR*n) x nb , (NR*n)

    # multivariate partial response: dlogM = sum_b beta_b * band_sigma_b
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # bootstrap over galaxies for errors
    NB = X.shape[1]; boots = np.zeros((300, NB))
    idx_all = np.arange(X.shape[0])
    for k in range(300):
        j = rng.integers(0, X.shape[0], X.shape[0])
        boots[k] = np.linalg.lstsq(X[j], y[j], rcond=None)[0]
    se = boots.std(0)

    print("\n" + "="*74)
    print("MASS-SELECTION BIAS vs DETECTION-BAND REST WAVELENGTH (JADES z~7-9)")
    print("  partial response d(logM)/d(band sigma), all bands degraded jointly, no selection\n")
    print(f"  {'band':7s} {'rest wav (A)':>12s}  {'d logM / dsigma':>16s}   sign")
    for i, b in enumerate(NIRCAM):
        rest = PIVOT[b]/(1+zmed)*1e4
        s = "buffers SMF (-)" if beta[i] < 0 else "inflates SMF (+)"
        print(f"  {b:7s} {rest:12.0f}  {beta[i]:+.4f} ± {se[i]:.4f}   {s}")
    # crossover -- restrict to bands ABOVE the Lyman break (rest > 1400 A); F090W/F115W are
    # Lyman-break dropout bands at z~8 and do not act as clean detection bands.
    rests = np.array([PIVOT[b]/(1+zmed)*1e4 for b in NIRCAM])
    clean = rests > 1400
    o = np.argsort(rests[clean]); br, bb = rests[clean][o], beta[clean][o]
    cross = None
    for i in range(len(bb)-1):
        if bb[i] < 0 <= bb[i+1] or bb[i] > 0 >= bb[i+1]:
            cross = br[i] + (br[i+1]-br[i])*(0-bb[i])/(bb[i+1]-bb[i]); break
    print(f"\n  => sign flip near rest {cross:.0f} A (F090W/F115W excluded: Lyman-break dropouts)"
          if cross else "\n  => no sign flip in clean range")
    print("     (the 4000A/Balmer break: UV-band selection buffers the SMF, optical-band inflates it)")
    print(f"  (elapsed {time.time()-t0:.0f}s)")
    save = os.environ.get("XDC_SAVE")
    if save:
        np.savez(save, bands=np.array(NIRCAM), rest=np.array([PIVOT[b]/(1+zmed)*1e4 for b in NIRCAM]),
                 beta=beta, se=se, zmed=zmed, crossover=(cross if cross else np.nan),
                 templ=os.path.basename(M.TEMPL))
        print(f"  saved -> {save}")


if __name__ == "__main__":
    main()
