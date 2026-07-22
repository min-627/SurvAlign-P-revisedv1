#!/bin/bash
# Part E: 다중 중첩 공격 환경에서 Survival Map(guide) 채널의 가치를 확인하는
# 채널 소거/하드마스킹 파일럿. 4개 조합을 순서대로 실행하고 각각의 로그를
# logs/part_e_pilot.log 에 이어붙인다(tee -a). RunPod에서 리포지토리 루트에서
# tmux 세션 안에 실행:
#
#   tmux new-session -s part_e_pilot
#   bash scripts/run_part_e_pilot.sh
#   (Ctrl+B, D 로 detach)
#
# 이후 다시 접속: tmux attach -t part_e_pilot
# 로그만 보고 싶으면: tail -f logs/part_e_pilot.log

set -e

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/part_e_pilot.log"
mkdir -p "${LOG_DIR}"
mkdir -p checkpoints results

run_combo() {
    local run_id="$1"
    shift
    echo "" | tee -a "${LOG_FILE}"
    echo "===== [${run_id}] START $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${LOG_FILE}"
    python phase2_training.py \
        --mode proposed_gate \
        --map_type survival \
        --dataset_type librispeech \
        --dataset_name dev-clean \
        --seed 42 \
        --epochs 5 \
        --train_cascade \
        --test_attacks clean,facodec \
        --strict_heldout \
        --val_max_batches -1 \
        --test_max_batches -1 \
        --results_dir "results/phase2_pilot_${run_id}" \
        --checkpoint_dir "checkpoints/phase2_pilot_${run_id}" \
        --run_id "${run_id}" \
        "$@" 2>&1 | tee -a "${LOG_FILE}"
    echo "===== [${run_id}] END   $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${LOG_FILE}"
}

# E1 [1,4] baseline: residual과 guide(Survival Map) 채널 둘 다 0으로 마스킹
run_combo "e1_no_residual_no_guide" --channel_ablation no_residual_no_guide

# E2 [1,2,4] 대조군: guide(Survival Map) 채널만 0으로 마스킹 (residual은 그대로)
run_combo "e2_no_guide" --channel_ablation no_guide

# E3 [1,3,4] 대체 + 하드마스킹: residual 채널만 0으로 마스킹, guide는 하드마스킹으로 강제 개입
run_combo "e3_no_residual_hardmask" --channel_ablation no_residual --hard_mask

# E4 [1,2,3,4] 전체 + 하드마스킹: 마스킹 없음(전 채널 사용) + guide 하드마스킹
run_combo "e4_full_hardmask" --channel_ablation full --hard_mask

echo "" | tee -a "${LOG_FILE}"
echo "===== Part E pilot: 4개 조합 전부 완료 $(date '+%Y-%m-%d %H:%M:%S') =====" | tee -a "${LOG_FILE}"
