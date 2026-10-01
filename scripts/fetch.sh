#!/usr/bin/env bash
# Fetch BAM (pinned) and the public raw servo logs into external/, then resample them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external/raw
[ -d external/bam ] || { git clone https://github.com/Rhoban/bam external/bam && git -C external/bam checkout -q e9a619d56da5236206f4de6ceec2c1ee1b497b5c; }
cd external/raw
HF=https://huggingface.co/buckets/Gregwar/bam_data/resolve
get() { [ -f "$2" ] || curl -sSL --fail -o "$2" "$1"; }
get "$HF/mx64_raw.tgz?download=true" mx64_raw.tgz
get "$HF/mx106_raw.tgz?download=true" mx106_raw.tgz
get "$HF/xl330_raw.zip?download=true" xl330_raw.zip
get "$HF/feetech_sts3215_raw.zip?download=true" sts3215_raw.zip
get https://github.com/T-K-233/bam/releases/download/sts3215-12v-bam-data-v1/feetech_sts3215_12v_raw.zip sts3215_12v_raw.zip
get https://github.com/i1Cps/duck_mini_pro_headless/releases/download/st3025-bam-data-v1/waveshare_st3025_raw.zip st3025_raw.zip
for f in mx64 mx106; do mkdir -p $f && tar xzf ${f}_raw.tgz -C $f; done
for f in xl330 sts3215 sts3215_12v st3025; do mkdir -p $f && unzip -q -o ${f}_raw.zip -d $f; done
cd ../..
python -m partgap.ingest
