#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=YOUR_CKPT_PATH
MODEL_BASE=YOUR_MODEL_BASE_PATH

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${CPULIST[$IDX]} python slowfocus/eval/model_fineaction_tsearch.py \
  --model-path ./work_dirs/$CKPT \
  --model-base $MODEL_BASE \
  --video_dir /cache/Finetune \
  --gt_file /cache/Eval/FineAction/fineaction_test.json \
  --output_dir ./work_dirs/eval_fineaction_tsearch/$CKPT \
  --output_name pred \
  --num-chunks $CHUNKS \
  --chunk-idx $IDX \
  --fps 1 \
  --max_frame_num 100 \
  --is_search 1 \
  --search dense_v0 \
  --search_sampling 20 \
  --is_prompt 1 \
  --conv_mode vicuna_v1 &

done

wait
echo "-------------------inference finished-------------------"

output_file=./work_dirs/eval_fineaction_tsearch/$CKPT/merge.json

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
  cat ./work_dirs/eval_fineaction_tsearch/$CKPT/${CHUNKS}_${IDX}.json >> "$output_file"
done
echo "-------------------merge finished--------------------"

source /etc/profile
python slowfocus/eval/eval_fineaction_tsearch.py \
--eval_file ./work_dirs/eval_fineaction_tsearch/$CKPT/merge.json
