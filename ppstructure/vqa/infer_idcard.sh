python infer_ser.py \
    --model_name_or_path "/mnt/ssd/marley/OCR/PaddleOCR/ppstructure/vqa/output/ser/best_model" \
    --ser_model_type "LayoutLMv2" \
    --output_dir "output/ser_shuffle/" \
    --infer_imgs "/mnt/ssd/marley/Fake-Data-Generator/result_data" \
    --ocr_json_path "/mnt/ssd/marley/OCR/LayoutXLM/paddle_ocr_idcard_data/test.json" \
    --label_path "/mnt/ssd/marley/OCR/LayoutXLM/paddle_ocr_idcard_data/labels.txt" \