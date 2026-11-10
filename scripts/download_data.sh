#!/usr/bin/env bash
# Download each site from Kaggle, cache it (crop + resize to 384 px JPEG), then delete the archive.
# One site at a time, so the disk never holds more than one large archive.
#
# Needs a Kaggle API token (KAGGLE_API_TOKEN, or ~/.kaggle/access_token) and the Kaggle CLI.
# Set KAGGLE to the command that runs it, e.g. KAGGLE="uvx --python 3.12 kaggle" (default: kaggle).
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
KAGGLE="${KAGGLE:-kaggle}"
ZIPS=data/zips
mkdir -p "$ZIPS"
if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "$HOME/.kaggle/access_token" ]]; then
    export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
fi

# site -> Kaggle dataset(s); archives are named after the dataset slug
declare -A DATASETS=(
    [messidor2]="mariaherrerot/messidor2preprocess google-brain/messidor2-dr-grades"
    [idrid]="aaryapatel98/indian-diabetic-retinopathy-image-dataset"
    [odir]="andrewmvd/ocular-disease-recognition-odir5k"
    [eyepacs]="tanlikesmath/diabetic-retinopathy-resized"
    [aptos]="mariaherrerot/aptos2019"
)

for site in messidor2 idrid odir eyepacs aptos; do
    if [[ -f "data/meta/$site.csv" ]]; then
        echo "$site: already cached"
        continue
    fi
    for dataset in ${DATASETS[$site]}; do
        archive="$ZIPS/${dataset#*/}.zip"
        [[ -f "$archive" ]] || $KAGGLE datasets download -d "$dataset" -p "$ZIPS" -q
    done
    "$PYTHON" scripts/prepare_data.py --site "$site"
    for dataset in ${DATASETS[$site]}; do
        rm -f "$ZIPS/${dataset#*/}.zip"
    done
done
