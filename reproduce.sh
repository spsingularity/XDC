#!/usr/bin/env bash
# Reproduce the XDC paper.
#
#   ./reproduce.sh          build figures from the committed results/ and compile the PDF
#                           (no data download, no eazy install required)
#   ./reproduce.sh all      rerun the full two-depth SED analysis to regenerate results/*.npz,
#                           then build figures + PDF (needs the JADES catalogue and eazy)
#
# Override the interpreter (e.g. a conda env that has eazy) with:
#   PYTHON=/path/to/python ./reproduce.sh all
# Override the catalogue path with XDC_CAT=/path/to/jades_dr5_gds_phot.fits
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
MODE="${1:-paper}"
CAT="${XDC_CAT:-data/jades_dr5_gds_phot.fits}"

if [ "$MODE" = "all" ]; then
  [ -f "$CAT" ] || { echo "ERROR: catalogue $CAT not found -- see data/README.md to download it."; exit 1; }
  echo ">> verifying catalogue checksum"
  ( cd data && shasum -a 256 -c CHECKSUMS.sha256 ) \
    || echo "WARNING: checksum mismatch -- catalogue differs from the one used in the paper; results may differ."
  EZ=$("$PYTHON" -c "import os,eazy;print(os.path.join(os.path.dirname(eazy.__file__),'data','eazy-photoz','templates'))")
  echo ">> [1/4] two-depth stellar-mass bias"
  XDC_SAVE=results/mass_matched.npz "$PYTHON" code/run_jades_mass_crossdepth.py "$CAT"
  echo ">> [2/4] selection-response spectrum"
  XDC_SAVE=results/spectrum.npz     "$PYTHON" code/run_jades_response_spectrum.py "$CAT"
  echo ">> [3/4] band response (FSPS templates)"
  XDC_SAVE=results/band_qsf.npz     "$PYTHON" code/run_jades_band_response.py "$CAT"
  echo ">> [4/4] band response (spline templates, robustness)"
  XDC_TEMPL="$EZ/spline_templates_v2/tweak_spline.param" XDC_SAVE=results/band_spline.npz \
    "$PYTHON" code/run_jades_band_response.py "$CAT"
elif [ "$MODE" != "paper" ]; then
  echo "usage: ./reproduce.sh [paper|all]"; exit 2
fi

echo ">> building figures from results/"
"$PYTHON" code/make_figures.py
echo ">> compiling paper"
( cd paper && pdflatex -interaction=nonstopmode xdc && bibtex xdc \
    && pdflatex -interaction=nonstopmode xdc && pdflatex -interaction=nonstopmode xdc ) >/dev/null
echo ">> done: paper/xdc.pdf"
