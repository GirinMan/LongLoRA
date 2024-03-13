current_time=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir outputs

accelerate launch fine-tune.py  \
        --model_name_or_path models/SOLAR-10.7B-dpo-v1 \
        --bf16 True \
        --output_dir outputs/opal-longlora-v0.3.GL.O/$current_time \
        --dataset_name legal_data/v0.3.GL.O \
        --model_max_length 4096 \
        --deepspeed "ds_configs/stage2.json" \
        --use_flash_attn True \
        --use_full_attn False \
        --low_rank_training True \
        --trainable_params embed,norm \
        --num_train_epochs 1  \
        --per_device_train_batch_size 8     \
        --per_device_eval_batch_size 8     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 200     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --tf32 True \
        --seed 42 \
        --do_eval True \
        --evaluation_strategy "steps" \
        --eval_steps 200 \
        --max_steps 2000 \
        2>&1 |tee train_${current_time}.log
