export HF_HOME=/home/girinman/.cache/
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nproc_per_node=4 --master_port 12345 supervised-fine-tune.py \
        --model_name_or_path /ckpt/BHSN-Legal-LLM/opal/BHSN-opal-003/v0.3.l.o_rank_8/iter_0000800 \
        --bf16 True \
        --output_dir adapters/20240409_longchecklist-16k_attention-s2_opal-1.0-240401 \
        --model_max_length 16384 \
        --use_flash_attn True \
        --use_full_attn False \
        --data_path dataset/longchecklist-16k/train_16k.json \
        --low_rank_training True \
        --num_train_epochs 3  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 1     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "no"     \
        --save_strategy "epoch"     \
        --save_total_limit 1     \
        --learning_rate 2e-4     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --tf32 True
