#!/bin/bash
# Before / after comparison of the level-2 trust test, on rorqual: the same twin study (11 problems x
# 8 starts x 3 seeds) and the real-data round trip (11 problems x staged / cma -> .CUL -> dscsm048), once
# with the legacy level-2 test (AJ_D31_TRUST=legacy), with the split one (exact + unrounded paths) and with
# the three-valued one (phase "three": exact path at three adjacent small steps, secants at 1 / 2 / 5 %).
# CMA-ES does not use the trust report: it runs once and is shared. Usage: d3_1_before_after.sh [phase]
set -u
DRV="python scripts/bench/calib/d3_1_staged.py"
D=$AGRI_JAX_DATA/validation/aj_d31_gradgap
L=$D/legacy; S=$D/split; V=$D/three
mkdir -p "$L" "$S" "$V"
phase=${1:-all}
if [ "$phase" = all ] || [ "$phase" = prep ]; then
  AJ_D31_DIR=$L $DRV prep --jobs 32 && AJ_D31_DIR=$L $DRV check || exit 1
  cp "$L"/inputs.pkl "$L"/inputs_meta.json "$L"/check_*.json "$S"/
fi
if [ "$phase" = all ] || [ "$phase" = runs ]; then
  for sd in 0 1 2; do
    AJ_D31_DIR=$L AJ_D31_TRUST=legacy $DRV runs --kind twin --methods staged,cma --variants base --seed $sd
    AJ_D31_DIR=$S AJ_D31_TRUST=split $DRV runs --kind twin --methods staged --variants base --seed $sd
    cp "$L"/twin_cma_base_s$sd.json "$S"/
  done
  AJ_D31_DIR=$L AJ_D31_TRUST=legacy $DRV runs --kind real --methods staged,cma --variants base --seed 0
  AJ_D31_DIR=$S AJ_D31_TRUST=split $DRV runs --kind real --methods staged --variants base --seed 0
  cp "$L"/real_cma_base_s0.json "$S"/
fi
if [ "$phase" = all ] || [ "$phase" = tables ]; then
  for t in legacy split; do
    d=$D/$t
    AJ_D31_DIR=$d $DRV roundtrip --kind real --method staged
    AJ_D31_DIR=$d $DRV roundtrip --kind real --method cma
    AJ_D31_DIR=$d $DRV table3 --methods staged,cma --variants base --seeds 0,1,2
    echo "=== $t"; cat "$d"/tables_twin3.md
  done
fi
if [ "$phase" = three ]; then
  cp "$L"/inputs.pkl "$L"/inputs_meta.json "$L"/check_*.json "$L"/twin_cma_base_s*.json "$L"/real_cma_base_s0.json "$V"/
  for sd in 0 1 2; do
    AJ_D31_DIR=$V AJ_D31_TRUST=three $DRV runs --kind twin --methods staged --variants base --seed $sd
  done
  AJ_D31_DIR=$V AJ_D31_TRUST=three $DRV runs --kind real --methods staged --variants base --seed 0
  AJ_D31_DIR=$V $DRV roundtrip --kind real --method staged
  AJ_D31_DIR=$V $DRV roundtrip --kind real --method cma
  AJ_D31_DIR=$V $DRV table3 --methods staged,cma --variants base --seeds 0,1,2
  echo "=== three"; cat "$V"/tables_twin3.md
fi
