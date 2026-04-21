#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install kindel

python3 -c "from kindel.utils.data import download_kindel; df = download_kindel('DDR1'); df.to_csv('data/kindel_ddr1.csv', index=False)"

echo "Wrote data/kindel_ddr1.csv"

