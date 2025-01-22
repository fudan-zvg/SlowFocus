#!/bin/bash

nproc_per_node=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_addr "$(echo ${VC_WORKER_HOSTS} | cut -d ',' -f 1)" --master_addr 56378 --node_rank=$VC_TASK_INDEX \
    slowfocus/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /cache/model_zoo/LLM/vicuna/vicuna-7b-v1.5 \
    --version plain_guided \
    --temporal_embedding v1 \
    --data_path /cache/Pretrain/llava_558k_with_webvid.json \
    --image_folder /cache/Pretrain/images \
    --video_folder /cache/Pretrain/videos \
    --vision_tower /cache/model_zoo/openai/clip-patch14-224 \
    --image_processor ./slowfocus/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bert_type "qformer_pretrain_freeze" \
    --num_query 32 \
    --pretrain_qformer /cache/model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \
    --compress_type "grid:8" \
    --bf16 False \
    --fp16 True \
    --output_dir ./work_dirs/slowfocus-7b-pretrain-224-video-fps-1-grid-8-clip-te-v1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
