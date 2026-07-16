#!/usr/bin/env python3
"""
XDC core: a MODEL-INDEPENDENT null test for correlated selection bias in the high-z SMF,
using survey DEPTH as an exclusion restriction (a variable that changes DETECTION but not
the true mass).

This version supersedes the original single-draw demo. It confronts the two confounds that
the first draft hid, and gives the estimator that survives them:

  DECOMPOSITION. The depth-dependent residual of a completeness-corrected SMF has THREE
  pieces (see paper Sec. "Three depth-dependent residuals"):
    (i)   volume incompleteness              -- removed by V/Vmax;
    (ii)  Eddington SCATTER bias             -- depends on the mass-error sigma, which is
          itself depth-dependent (deeper -> higher S/N -> smaller sigma); NOT removed by
          V/Vmax and NOT proportional to rho;
    (iii) correlated SELECTION bias          -- the rho term the test targets.
  A naive V/Vmax cross-depth ratio conflates (ii) and (iii): a depth-dependent sigma alone
  fakes a signal as large as rho=0.6 (Part A).

  FIX 1 (binned, needs per-tier sigma only): NOISE-MATCHING. Degrade the deep tier's inferred
  masses to the shallow tier's sigma. This equalizes kernel (ii) across depth so it cancels
  in the ratio, while the selection-tied term (iii) does not. Removes the confound (Part B).

  FIX 2 (fully model-independent): MATCHED-OVERLAP. On sky imaged at two depths, for objects
  detected in BOTH tiers compare the inferred mass shallow-minus-deep. The true mass AND its
  scatter convolution cancel object-by-object, so no sigma or completeness model is needed;
  a nonzero mean shift is a direct, depth-differential measurement of rho. This has power at
  CURRENT JADES volume, unlike the massive-end binned ratio (Part C).

Run: python3 test_crossdepth_nulltest.py
"""
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(7)

alpha, logMstar, phistar = -1.85, 10.72, 1.3e-5     # z~8 Schechter (Stefanon+2021)
A_SEL = 2.5                                          # selection sharpness d(index)/d(logM)
edges = np.arange(8.5, 12.01, 0.05); cc = 0.5*(edges[:-1]+edges[1:]); dcc = np.diff(edges)
rep_e = np.arange(10.0, 11.5, 0.3); rep = 0.5*(rep_e[:-1]+rep_e[1:])
def schechter(lm):
    x = 10**(lm-logMstar); return np.log(10)*phistar*x**(alpha+1)*np.exp(-x)


def one_field(V, lim, sigma, rho, degrade_to=None):
    """One depth tier. Returns (density, counts, weighted_var) on rep bins.
    lim = mass-completeness limit (sets depth); degrade_to = target sigma for noise-matching."""
    k = rng.poisson(schechter(cc)*dcc*V)
    lmT = np.repeat(cc, k)
    e = rng.multivariate_normal([0, 0], [[1, rho*sigma], [rho*sigma, sigma**2]], lmT.size)
    s = A_SEL*(lmT - lim) + e[:, 0]                  # latent detectability index
    det = rng.random(lmT.size) < norm.cdf(s)
    lmO = lmT + e[:, 1]
    if degrade_to is not None and degrade_to > sigma:
        lmO = lmO + rng.normal(0, np.sqrt(degrade_to**2 - sigma**2), lmT.size)  # noise-match
    pdet = norm.cdf(s)
    ic = np.clip(np.digitize(lmO, rep_e)-1, 0, len(rep)-1)
    comp = np.array([np.clip(pdet[det][ic[det] == i].mean() if (ic[det] == i).any() else 1,
                             .05, 1) for i in range(len(rep))])
    ib = np.clip(np.digitize(lmO[det], rep_e)-1, 0, len(rep)-1)
    w = 1.0/comp[ib]
    h, _ = np.histogram(lmO[det], bins=rep_e, weights=w)
    w2, _ = np.histogram(lmO[det], bins=rep_e, weights=w**2)
    cnt, _ = np.histogram(lmO[det], bins=rep_e)
    dm = np.diff(rep_e)*V
    return h/dm, cnt, np.sqrt(w2)/dm                 # density, counts, density error (sqrt Sum w^2)


def binned_ratio(V, rho, sig_d, sig_s, noise_match, lim_d=9.0, lim_s=9.8, nreal=40):
    """Median deep/shallow ratio (+MAD) over realizations at the massive end."""
    tgt = max(sig_d, sig_s) if noise_match else None
    R = np.full((nreal, len(rep)), np.nan)
    for r in range(nreal):
        d, nd, ed = one_field(V, lim_d, sig_d, rho, degrade_to=tgt)
        s, ns, es = one_field(V, lim_s, sig_s, rho)
        ok = (nd >= 3) & (ns >= 3) & (s > 0)
        R[r, ok] = d[ok]/s[ok]
    return np.nanmedian(R, 0), np.nanmedian(np.abs(R-np.nanmedian(R, 0)), 0)*1.4826


def matched_overlap(V, rho, sig_d, sig_s, lim_d=8.5, lim_s=9.5, nreal=300):
    """Mean shallow-minus-deep inferred-mass shift among jointly-detected objects.
    Fully model-independent: true mass cancels object-by-object. Returns per-realization
    (mean shift, its s.e., N_pairs), then detection fraction at 3 sigma."""
    means, ses, npair = [], [], []
    for _ in range(nreal):
        k = rng.poisson(schechter(cc)*dcc*V); lmT = np.repeat(cc, k); N = lmT.size
        ed = rng.multivariate_normal([0, 0], [[1, rho*sig_d], [rho*sig_d, sig_d**2]], N)
        es = rng.multivariate_normal([0, 0], [[1, rho*sig_s], [rho*sig_s, sig_s**2]], N)
        det_d = rng.random(N) < norm.cdf(A_SEL*(lmT-lim_d) + ed[:, 0])
        det_s = rng.random(N) < norm.cdf(A_SEL*(lmT-lim_s) + es[:, 0])
        both = det_d & det_s
        if both.sum() < 5:
            continue
        dM = (lmT+es[:, 1])[both] - (lmT+ed[:, 1])[both]   # shallow_obs - deep_obs
        means.append(dM.mean()); ses.append(dM.std()/np.sqrt(both.sum())); npair.append(both.sum())
    means, ses, npair = map(np.array, (means, ses, npair))
    detrate = np.mean(np.abs(means) > 3*ses)
    return means.mean(), np.median(ses), int(np.median(npair)), detrate


Vnow, Vfut = 2.5e5, 2.5e7      # current JADES-like ; Roman/Euclid-like (100x)

print("="*78)
print("PART A -- the confound: naive V/Vmax cross-depth ratio (Roman/Euclid volume)")
print("  A depth-dependent mass-error sigma fakes a signal even at rho=0.")
print("   case                          " + "  ".join(f">{rep_e[i]:.1f}" for i in range(len(rep))))
for lbl, sd, ss, rho in [("rho=0   sigma 0.25/0.25 (equal) ", .25, .25, 0.),
                         ("rho=0   sigma 0.15/0.35 (depth!)", .15, .35, 0.),
                         ("rho=0.6 sigma 0.25/0.25 (signal)", .25, .25, .6)]:
    med, _ = binned_ratio(Vfut, rho, sd, ss, noise_match=False)
    print(f"   {lbl}   " + "  ".join(f"{v:4.2f}" if np.isfinite(v) else "  - " for v in med))

print("\n" + "="*78)
print("PART A2 -- confound: the empirical completeness estimator, evaluated NEAR the limit")
print("  Report bins sitting ON the completeness limit (lim 10.0/10.6) bias rho=0 away from 1")
print("  -- this is why the first-draft single-draw Table 1 (rho=0.6->1.9) was contaminated.")
print("   case                          " + "  ".join(f">{rep_e[i]:.1f}" for i in range(len(rep))))
for lbl, sd, ss, rho in [("rho=0   sigma equal, report~limit", .25, .25, 0.)]:
    med, _ = binned_ratio(Vfut, rho, sd, ss, noise_match=False, lim_d=10.0, lim_s=10.6)
    print(f"   {lbl}   " + "  ".join(f"{v:4.2f}" if np.isfinite(v) else "  - " for v in med))
print("  => evaluate the binned ratio only SAFELY ABOVE the shallow limit (Parts A/B do).")

print("\n" + "="*78)
print("PART B -- FIX 1: noise-matching (degrade deep to shallow sigma). Same cases.")
print("  The sigma(depth) confound cancels; the rho signal survives.")
print("   case                          " + "  ".join(f">{rep_e[i]:.1f}" for i in range(len(rep))))
for lbl, sd, ss, rho in [("rho=0   sigma 0.15/0.35 (depth!)", .15, .35, 0.),
                         ("rho=0.6 sigma 0.15/0.35 (signal)", .15, .35, .6),
                         ("rho=0.6 sigma 0.25/0.25 (signal)", .25, .25, .6)]:
    med, _ = binned_ratio(Vfut, rho, sd, ss, noise_match=True)
    print(f"   {lbl}   " + "  ".join(f"{v:4.2f}" if np.isfinite(v) else "  - " for v in med))

print("\n" + "="*78)
print("PART C -- FIX 2: matched-overlap shift <ΔlogM> = shallow - deep (model-independent)")
print("  Powerful already at CURRENT volume, because it pools ALL jointly-detected objects.")
print(f"   {'case':30s} {'volume':8s}  <Δm> (dex)      N_pairs  det.rate(3σ)")
for lbl, sd, ss, rho, V in [
        ("rho=0   (null)                ", .25, .25, 0.,  Vnow),
        ("rho=0.6 (bias present)        ", .25, .25, .6,  Vnow),
        ("rho=0.6 unequal sigma         ", .15, .35, .6,  Vnow),
        ("rho=0   (null)                ", .25, .25, 0.,  Vfut),
        ("rho=0.6 (bias present)        ", .25, .25, .6,  Vfut)]:
    m, se, npr, dr = matched_overlap(V, rho, sd, ss)
    tag = "current" if V == Vnow else "future"
    print(f"   {lbl} {tag:8s}  {m:+.4f} ± {se:.4f}   {npr:6d}   {dr*100:4.0f}%")

print("""
VERDICT
  * The binned cross-depth ratio has THREE depth-dependent biases, not one: (A) a
    depth-dependent scatter sigma(depth) fakes a signal as large as rho=0.6; (A2) the
    empirical completeness estimator biases the ratio near the limit -- so the first-draft
    single-draw Table 1 (rho=0.6->1.9) conflated all three, not a clean rho measurement.
  * The sigma(depth) confound is removed by noise-matching (per-tier sigma only, no rho
    model, Part B); the completeness artifact is avoided by reporting safely above the limit.
    Done right, the binned null is clean -- but a WEAK detector of rho above the limit.
  * The matched-overlap shift is fully model-independent (true mass cancels object-by-object).
    Per single JADES-like overlap it is already ~2 sigma sensitive to rho=0.6 (Part C: shift
    ~0.08-0.13 dex vs s.e. ~0.05), where the massive-end binned ratio is Poisson-limited;
    combining overlap fields or future volume makes it decisive (100%). It is the recommended
    real-data estimator on two-depth overlaps.""")
