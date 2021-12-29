import json
from PIL import Image, ImageDraw, ImageFont
import os

def draw_box_txt(draw, bbox, label, font, color='red'):
    draw.rectangle(bbox, fill=color)

    size = font.getsize(label)
    start_y = max(0, bbox[1] - size[1])
    draw.rectangle([bbox[0], start_y, bbox[0] + size[0], start_y + size[1]], fill=color)
    draw.text((bbox[0], start_y), label, fill='black', font=font)

image_dir = '/mnt/ssd/marley/Fake-Data-Generator/result_data'

with open('/mnt/ssd/marley/OCR/LayoutXLM/paddle_ocr_idcard_data/train.json', 'r') as f:
    lines = f.readlines()

font = ImageFont.load_default()
for line in lines[:10]:
    image_name, info_str = line.split("\t")
    class_type = '_'.join(image_name.split('_')[:-1])
    image = Image.open(os.path.join(image_dir, class_type, image_name))
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)
    ocr_info = json.loads(info_str)["ocr_info"]
    for item in ocr_info:
        if item['label'] == 'chxh':
            print(item['bbox'])
        bbox = item['bbox']
        label = item['label']
        draw_box_txt(draw, bbox, label, font)
    img_new = Image.blend(image, img_new, 0.3)
    img_new.save(image_name)
