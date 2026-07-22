#!/bin/bash
# Big Gate 용량 확장 실험: 오늘 Part E 파일럿(e1~e6)에서 쓴 것과 정확히 동일한
# --channel_ablation/--map_type/--hard_mask 조합에, Gate 용량만 키운
# --gate_hidden_dim 64 --gate_extra_layer를 추가해 BG1~BG6으로 재실행한다.
# "정보 충돌"(e4/e5=0.365 < e6=0.422)이 (a) 정보 자체의 본질적 충돌인지,
# (b) 얕은 Gate(hidden=16, 3층)의 용량 부족인지 분리하기 위함.
#
# RunPod에서 리포지토리 루트에서 tmux 세션 안에 실행:
#
#   tmux new-session -s big_gate_pilot
#   bash scripts/run_big_gate_pilot.sh
#   (Ctrl+B, D 로 detach)
#
# 이후 다시 접속: tmux attach -t big_gate_pilot
# 로그만 보고 싶으면: tail -f logs/big_gate_pilot.log

set -e

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/big_gate_pilot.log"
mkdir -p "${LOG_DIR}"
mkdir -p checkpoints results

run_combo() {
    local run_id="$1"
    shift
    echo "" | tee -a "${LOG_FILE}"
    echo "===== [${run_id}] START $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${LOG_FILE}"
    python phase2_training.py \
        --mode proposed_gate \
        --dataset_type librispeech \
        --dataset_name dev-clean \
        --seed 42 \
        --epochs 5 \
        --train_cascade \
        --test_attacks clean,facodec \
        --strict_heldout \
        --val_max_batches -1 \
        --test_max_batches -1 \
        --gate_hidden_dim 64 \
        --gate_extra_layer \
        --results_dir "results/phase2_pilot_${run_id}" \
        --checkpoint_dir "checkpoints/phase2_pilot_${run_id}" \
        --run_id "${run_id}" \
        "$@" 2>&1 | tee -a "${LOG_FILE}"
    echo "===== [${run_id}] END   $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${LOG_FILE}"
}

# BG1 (= e1) [1,4] baseline: residual과 guide 채널 둘 다 0으로 마스킹
run_combo "bg1_no_residual_no_guide" --channel_ablation no_residual_no_guide

# BG2 (= e2) [1,2,4] 대조군(기존 방식): guide 채널만 0으로 마스킹
run_combo "bg2_no_guide" --channel_ablation no_guide

# BG3 (= e3) [1,3,4] survival + 하드마스킹: residual만 마스킹, guide는 하드마스킹으로 강제 개입
run_combo "bg3_no_residual_hardmask" --map_type survival --channel_ablation no_residual --hard_mask

# BG4 (= e4) [1,2,3,4] survival + 하드마스킹: 마스킹 없음(전 채널) + guide 하드마스킹
run_combo "bg4_full_hardmask" --map_type survival --channel_ablation full --hard_mask

# BG5 (= e5) [1,2,3,4] survival, soft(하드마스킹 없음): 오늘 가장 저조했던 조합의 Big Gate 버전
#            -- 용량이 정보충돌을 해소하는지 확인하는 핵심 비교
run_combo "bg5_full_soft" --map_type survival --channel_ablation full

# BG6 (= e6) [1,3,4] survival, soft(하드마스킹 없음): 오늘의 최고 조합(e6=0.422)의 Big Gate 버전
run_combo "bg6_no_residual_soft" --map_type survival --channel_ablation no_residual

echo "" | tee -a "${LOG_FILE}"
echo "===== Big Gate pilot: 6개 조합 전부 완료 $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${LOG_FILE}"
