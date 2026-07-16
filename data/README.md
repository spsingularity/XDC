# Data

The analysis reads one public JWST catalogue that is **not** stored in this repository (it is
large and freely available). Download it into this directory, or pass its path as the first
argument to each `run_jades_*.py` script. Data files (`*.fits`) are gitignored.

## JADES DR5 GOODS-S photometry (~6.2 GB) — for `run_jades_*.py`
EAZY photo-z (`z_peak`, `Prob_gt_N`), per-band CIRC fluxes (nJy), and per-band inverse-variance
weight maps (`*_WHT`, the depth proxy). Stellar masses and the other SED quantities are refit in
this repository with `eazy`; no separate value-added mass catalogue is required.

- Primary: <https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/catalogs/hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits>
- Mirror (MAST): <https://archive.stsci.edu/hlsp/jades>

After downloading, verify you have the exact catalogue used in the paper (JADES may update it):
```sh
cd data && shasum -a 256 -c CHECKSUMS.sha256
```

## eazy templates and filter curves
The SED-refitting runners use [`eazy`](https://github.com/gbrammer/eazy-py). Its FSPS/spline
templates and `FILTER.RES` filter curves are fetched once with
`python3 -c "import eazy; eazy.fetch_eazy_photoz()"` (see the top-level README) and are not stored
here.
