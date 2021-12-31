python train_ser_idcard.py \
    --model_name_or_path "layoutlmv2-base-uncased" \
    --ser_model_type "LayoutLMv2" \
    --image_data_dir "/mnt/ssd/marley/Fake-Data-Generator/result_data" \
    --train_label_path "/mnt/ssd/marley/OCR/LayoutXLM/paddle_ocr_idcard_data_v2/train.json" \
    --eval_label_path "/mnt/ssd/marley/OCR/LayoutXLM/paddle_ocr_idcard_data_v2/val.json" \
    --num_train_epochs 25 \
    --eval_steps 1000 \
    --output_dir "./output/ser_1231/" \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --evaluate_during_training \
    --seed 2048 \
    --print_step 100 \
    --per_gpu_eval_batch_size 8 \
    --per_gpu_train_batch_size 8 \
    --label_path "/mnt/ssd/marley/OCR/LayoutXLM/paddle_ocr_idcard_data_v2/labels.txt" \
    --online_augment