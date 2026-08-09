#!/usr/bin/env bash
# =============================================================================
#  REPRODUCE TODO EL ESTUDIO DE INCERTIDUMBRE DINAMICA / GRAY-BOX
# =============================================================================
#
#  Corre las fases F0..F7 completas y deja los resultados en results/uncertainty
#  y las figuras en results/figures. Ver docs/graybox_manual_completo.md.
#
#  PARALELISMO: 4 procesos x 2 hilos = 8 nucleos. Medido: mas hilos por proceso
#  NO acelera (satura en 2), asi que conviene paralelizar entre corridas y no
#  dentro de cada una.
#
#  DURACION aproximada en una maquina de 8 nucleos: ~2.5 h.
#  (Los entrenamientos con correccion neuronal son ~3x mas lentos que el
#   white-box, porque g_phi se evalua en cada subpaso del RK4 del rollout.)
#
#  USO:  bash scripts/run_uncertainty_all.sh
# =============================================================================
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs results/uncertainty results/figures

ola() {
  echo "### ola de $# corridas — $(date +%H:%M:%S)"
  for c in "$@"; do eval "$c" & done
  wait
  echo "### ola terminada — $(date +%H:%M:%S)"
}

echo "=== F0 · tests de la infraestructura ==="
$PY -m pytest tests/test_uncertainty.py tests/test_graybox.py -q

echo
echo "=== F1 · generar los datasets del barrido de eps y caracterizarlos ==="
$PY scripts/gen_uncertain_dataset.py
$PY scripts/exp_f1_characterize.py

echo
echo "=== F4b · geometria del mismatch (barato, y explica todo lo demas) ==="
$PY scripts/exp_f4b_geometria_mismatch.py

echo
echo "=== F2 · costo de la rigidez: white-box sobre datos perturbados ==="
ola "$PY scripts/exp_f2_rigidity_cost.py --eps 0    > logs/f2_eps0.log   2>&1" \
    "$PY scripts/exp_f2_rigidity_cost.py --eps 0.25 > logs/f2_eps0.25.log 2>&1" \
    "$PY scripts/exp_f2_rigidity_cost.py --eps 0.5  > logs/f2_eps0.5.log  2>&1" \
    "$PY scripts/exp_f2_rigidity_cost.py --eps 1    > logs/f2_eps1.log    2>&1"
ola "$PY scripts/exp_f2_rigidity_cost.py --eps 1.5  > logs/f2_eps1.5.log  2>&1" \
    "$PY scripts/exp_f2_rigidity_cost.py --eps 2    > logs/f2_eps2.log    2>&1"

echo
echo "=== F3 · variantes gray-box con mismatch (eps=1) ==="
ola "$PY scripts/exp_f3_graybox.py --variant A --eps 1          > logs/f3_A_eps1.log 2>&1" \
    "$PY scripts/exp_f3_graybox.py --variant B --eps 1          > logs/f3_B_eps1.log 2>&1" \
    "$PY scripts/exp_f3_graybox.py --variant C --eps 1 --lam 0.1 > logs/f3_C_l0.1.log 2>&1" \
    "$PY scripts/exp_f3_graybox.py --variant C --eps 1 --lam 1   > logs/f3_C_l1.log   2>&1"
ola "$PY scripts/exp_f3_graybox.py --variant D --eps 1 --lam 1   > logs/f3_D_l1.log   2>&1" \
    "$PY scripts/exp_f3_graybox.py --variant D --eps 1 --lam 10  > logs/f3_D_l10.log  2>&1" \
    "$PY scripts/exp_f3_graybox.py --variant A --eps 0          > logs/f3_A_eps0.log 2>&1" \
    "$PY scripts/exp_f3_graybox.py --variant B --eps 0          > logs/f3_B_eps0.log 2>&1"
#   ^ las dos ultimas son EL CONTROL que decide la interpretacion: sin hueco
#     estructural la correccion no tiene fisica legitima que explicar, asi que
#     solo puede competir con los parametros. Ver seccion 15.0 del manual.

echo
echo "=== F4 · FIM del hibrido · F7 · controles ==="
ola "$PY scripts/exp_f4_fim_hybrid.py --data data/processed/uncertain/eps1.npz \
        --ckpt results/uncertainty/models/f3_B_eps1.pt > logs/f4_fim_B.log 2>&1" \
    "$PY scripts/exp_f7_controls.py > logs/f7_controls.log 2>&1"

echo
echo "=== F5 · recuperacion funcional ==="
for t in A_eps1 B_eps1 C_eps1_lam1 D_eps1_lam1 D_eps1_lam10; do
  $PY scripts/exp_f5_functional_recovery.py \
      --ckpt "results/uncertainty/models/f3_${t}.pt" > "logs/f5_${t}.log" 2>&1 || true
done

echo
echo "=== F6 · lazo cerrado ==="
$PY scripts/exp_f6_closed_loop.py --eps 1.0 \
    --graybox results/uncertainty/models/f3_D_eps1_lam1.pt

echo
echo "=== INFORME CONSOLIDADO ==="
$PY scripts/informe_incertidumbre.py | tee results/uncertainty/informe.txt

echo
echo "=== TODO COMPLETO — $(date +%H:%M:%S) ==="
