# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import cv2
import random
import numpy as np
import imghdr
from copy import deepcopy
import re 

import paddle

from PIL import Image, ImageDraw, ImageFont

ALL_EXCLUDE_FIELD = set(['key_ngon_tro_trai', 'key_ngon_tro_phai', 'key_csgt', 'key_title', 'key_title_en', \
        'key_chung_nhan', 'key_chung_nhan_1_en', 'key_chung_nhan_2_en', 'key_chung_nhan_3_en', 'note', 'so_serial', 'random_number', 'tires', 'level_nguoi_cap', \
        'address_date', 'nguoi_cap', 'quy_dinh_them', 'bo_giao_thong', 'cuc_dan_kiem_vn', 'cuc_dan_kiem_vn_en', 'commercial', 'modification', 'equipment', 'type_fuel', \
        'inspection_stamp', 'van_tay_trai', 'van_tay_phai', 'driver_license_bo_gtvt', 'chxh', 'title', 'ignore', 'chxh_en', 'cong_an', 'cong_an_thanh_pho_en', 'wheel_formula', \
        'wheel_tread', 'container_dim', 'wheelbase', 'kerb_mass', 'pay_load', 'total_mass', 'towed_mass', 'engine_displace', 'max_rpm', 'inspection_report', 'cong_an_quan', \
        'cong_an_quan_en', 'so_cho_nam', 'so_cho_dung', 'cong_suat', 'cong_an_thanh_pho', 'total_mass', 'so_cho_nam', 'chieu_dai', 'chieu_rong', 'chieu_cao', 'overall_dim', \
        'chxh', 'signature', 'stamp_circle', 'bo_gtvt', 'bo_gtvt_en', 'title_en','key_chieu_cao', 'key_chieu_dai', 'key_chieu_rong', 'key_commercial', 'key_cong_suat', \
        'key_container_dim', 'key_engine_displace', 'key_equipment', 'key_hang_hoa', 'key_ho_ten_khac', 'key_inspection_center', 'key_inspection_report', \
        'key_inspection_stamp', 'key_kich_thuoc_bao', 'key_max_rpm','key_modification', 'key_note', 'key_overall_dim', 'key_pay_load', 'key_san_xuat', 'key_so_cho_dung', \
        'key_so_cho_nam', 'key_so_serial', 'key_specification', 'key_tai_trong', 'key_ten_dong_co', 'key_tires', 'key_total_mass', 'key_towed_mass', 'key_type_fuel', \
        'key_vehicle', 'key_wheel_formula', 'key_wheel_tread', 'key_wheelbase', 'nuoc_san_xuat'])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def get_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    if "O" not in lines:
        lines.insert(0, "O")
    labels = []
    for line in lines:
        if line == "O":
            labels.append("O")
        else:
            labels.append("B-" + line)
            labels.append("I-" + line)
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def get_bio_label_maps_idcard(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    labels = []
    entities_labels = []
    labels.append('O')
    entities_labels.append('loai_xe')
    labels.append('B-loai_xe')
    labels.append('I-loai_xe')
    for line in lines:
        line = re.sub(r"_\d+$", "", line).lower()
        if line.isdigit() \
            or line.startswith('en_') \
                or line.startswith('unit_') \
                    or line.startswith('tires_') \
                        or line.endswith('_en') \
                            or line.endswith('_ta') \
                                or line.startswith('loai_xe') \
                                    or (line.startswith('hang') and line != 'hang_cap') \
                                        or line in ALL_EXCLUDE_FIELD:
            continue
        else:
            entities_labels.append(line)
            labels.append('B-'+line)
            labels.append('I-'+line)

    labels = list(set(labels))
    labels.sort()
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    entities_labels = list(set(entities_labels))
    entities_labels.sort()
    entities_label2id_map = {label: idx for idx, label in enumerate(entities_labels)}
    
    return label2id_map, id2label_map, entities_label2id_map


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def draw_ser_results(image,
                     ocr_results,
                     font_path="../../doc/fonts/simfang.ttf",
                     font_size=18):
    np.random.seed(2021)
    color = (np.random.permutation(range(255)),
             np.random.permutation(range(255)),
             np.random.permutation(range(255)))
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx])
        for idx in range(1, 255)
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    for ocr_info in ocr_results:
        if ocr_info["pred_id"] not in color_map:
            continue
        color = color_map[ocr_info["pred_id"]]
        # text = "{}: {}".format(ocr_info["pred"], ocr_info["text"])
        text = ocr_info["pred"]

        draw_box_txt(ocr_info["bbox"], text, draw, font, font_size, color)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # draw ocr results
    start_y = max(0, bbox[0][1] - font_size)
    tw = font.getsize(text)[0]
    draw.rectangle(
        [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + font_size)],
        fill=(0, 0, 255))
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)


def draw_re_results(image,
                    result,
                    font_path="../../doc/fonts/simfang.ttf",
                    font_size=18):
    np.random.seed(0)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    color_head = (0, 0, 255)
    color_tail = (255, 0, 0)
    color_line = (0, 255, 0)

    for ocr_info_head, ocr_info_tail in result:
        draw_box_txt(ocr_info_head["bbox"], ocr_info_head["text"], draw, font,
                     font_size, color_head)
        draw_box_txt(ocr_info_tail["bbox"], ocr_info_tail["text"], draw, font,
                     font_size, color_tail)

        center_head = (
            (ocr_info_head['bbox'][0] + ocr_info_head['bbox'][2]) // 2,
            (ocr_info_head['bbox'][1] + ocr_info_head['bbox'][3]) // 2)
        center_tail = (
            (ocr_info_tail['bbox'][0] + ocr_info_tail['bbox'][2]) // 2,
            (ocr_info_tail['bbox'][1] + ocr_info_tail['bbox'][3]) // 2)

        draw.line([center_head, center_tail], fill=color_line, width=5)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


# pad sentences
def pad_sentences(tokenizer,
                  encoded_inputs,
                  max_seq_len=512,
                  pad_to_max_seq_len=True,
                  return_attention_mask=True,
                  return_token_type_ids=True,
                  return_overflowing_tokens=False,
                  return_special_tokens_mask=False):
    # Padding with larger size, reshape is carried out
    max_seq_len = (
        len(encoded_inputs["input_ids"]) // max_seq_len + 1) * max_seq_len

    needs_to_be_padded = pad_to_max_seq_len and \
        max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

    if needs_to_be_padded:
        difference = max_seq_len - len(encoded_inputs["input_ids"])
        if tokenizer.padding_side == 'right':
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"]) + [0] * difference
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] +
                    [tokenizer.pad_token_type_id] * difference)
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs[
                    "special_tokens_mask"] + [1] * difference
            encoded_inputs["input_ids"] = encoded_inputs[
                "input_ids"] + [tokenizer.pad_token_id] * difference
            encoded_inputs["bbox"] = encoded_inputs["bbox"] + [[0, 0, 0, 0]
                                                               ] * difference
    else:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                "input_ids"])

    return encoded_inputs


def split_page(encoded_inputs, max_seq_len=512):
    """
    truncate is often used in training process
    """
    for key in encoded_inputs:
        if key == 'entities':
            encoded_inputs[key] = [encoded_inputs[key]]
            continue
        encoded_inputs[key] = paddle.to_tensor(encoded_inputs[key])
        if encoded_inputs[key].ndim <= 1:  # for input_ids, att_mask and so on
            encoded_inputs[key] = encoded_inputs[key].reshape([-1, max_seq_len])
        else:  # for bbox
            encoded_inputs[key] = encoded_inputs[key].reshape(
                [-1, max_seq_len, 4])
    return encoded_inputs


def preprocess(
        tokenizer,
        ori_img,
        ocr_info,
        img_size=(224, 224),
        pad_token_label_id=-100,
        max_seq_len=512,
        add_special_ids=False,
        return_attention_mask=True, ):
    ocr_info = deepcopy(ocr_info)
    height = ori_img.shape[0]
    width = ori_img.shape[1]

    img = cv2.resize(ori_img, img_size).transpose([2, 0, 1]).astype(np.float32)

    segment_offset_id = []
    words_list = []
    bbox_list = []
    input_ids_list = []
    token_type_ids_list = []
    entities = []

    for info in ocr_info:
        # x1, y1, x2, y2
        bbox = info["bbox"]
        bbox[0] = int(bbox[0] * 1000.0 / width)
        bbox[2] = int(bbox[2] * 1000.0 / width)
        bbox[1] = int(bbox[1] * 1000.0 / height)
        bbox[3] = int(bbox[3] * 1000.0 / height)

        text = info["text"]
        encode_res = tokenizer.encode(
            text, pad_to_max_seq_len=False, return_attention_mask=True)

        if not add_special_ids:
            # TODO: use tok.all_special_ids to remove
            encode_res["input_ids"] = encode_res["input_ids"][1:-1]
            encode_res["token_type_ids"] = encode_res["token_type_ids"][1:-1]
            encode_res["attention_mask"] = encode_res["attention_mask"][1:-1]

        # for re
        entities.append({
            "start": len(input_ids_list),
            "end": len(input_ids_list) + len(encode_res["input_ids"]),
            "label": "O",
        })

        input_ids_list.extend(encode_res["input_ids"])
        token_type_ids_list.extend(encode_res["token_type_ids"])
        bbox_list.extend([bbox] * len(encode_res["input_ids"]))
        words_list.append(text)
        segment_offset_id.append(len(input_ids_list))

    encoded_inputs = {
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
        "bbox": bbox_list,
        "attention_mask": [1] * len(input_ids_list),
        "entities": entities
    }

    encoded_inputs = pad_sentences(
        tokenizer,
        encoded_inputs,
        max_seq_len=max_seq_len,
        return_attention_mask=return_attention_mask)

    encoded_inputs = split_page(encoded_inputs)

    fake_bs = encoded_inputs["input_ids"].shape[0]

    encoded_inputs["image"] = paddle.to_tensor(img).unsqueeze(0).expand(
        [fake_bs] + list(img.shape))

    encoded_inputs["segment_offset_id"] = segment_offset_id

    return encoded_inputs


def postprocess(attention_mask, preds, id2label_map):
    if isinstance(preds, paddle.Tensor):
        preds = preds.numpy()
    preds = np.argmax(preds, axis=2)

    preds_list = [[] for _ in range(preds.shape[0])]

    # keep batch info
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if attention_mask[i][j] == 1:
                preds_list[i].append(id2label_map[preds[i][j]])

    return preds_list


def merge_preds_list_with_ocr_info(ocr_info, segment_offset_id, preds_list,
                                   label2id_map_for_draw):
    # must ensure the preds_list is generated from the same image
    preds = [p for pred in preds_list for p in pred]

    id2label_map = dict()
    for key in label2id_map_for_draw:
        val = label2id_map_for_draw[key]
        if key == "O":
            id2label_map[val] = key
        if key.startswith("B-") or key.startswith("I-"):
            id2label_map[val] = key[2:]
        else:
            id2label_map[val] = key

    for idx in range(len(segment_offset_id)):
        if idx == 0:
            start_id = 0
        else:
            start_id = segment_offset_id[idx - 1]

        end_id = segment_offset_id[idx]

        curr_pred = preds[start_id:end_id]
        curr_pred = [label2id_map_for_draw[p] for p in curr_pred]

        if len(curr_pred) <= 0:
            pred_id = 0
        else:
            counts = np.bincount(curr_pred)
            pred_id = np.argmax(counts)
        ocr_info[idx]["pred_id"] = int(pred_id)
        ocr_info[idx]["pred"] = id2label_map[int(pred_id)]
    return ocr_info


def print_arguments(args, logger=None):
    print_func = logger.info if logger is not None else print
    """print arguments"""
    print_func('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print_func('%s: %s' % (arg, value))
    print_func('------------------------------------------------')


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # yapf: disable
    parser.add_argument("--model_name_or_path",
                        default=None, type=str, required=True,)
    parser.add_argument("--ser_model_type",
                        default='LayoutXLM', type=str)
    parser.add_argument("--re_model_name_or_path",
                        default=None, type=str, required=False,)
    parser.add_argument("--train_data_dir", default=None,
                        type=str, required=False,)
    parser.add_argument("--train_label_path", default=None,
                        type=str, required=False,)
    parser.add_argument("--eval_data_dir", default=None,
                        type=str, required=False,)
    parser.add_argument("--eval_label_path", default=None,
                        type=str, required=False,)
    parser.add_argument("--output_dir", default=None, type=str, required=True,)
    parser.add_argument("--max_seq_length", default=512, type=int,)
    parser.add_argument("--evaluate_during_training", action="store_true",)
    parser.add_argument("--num_workers", default=8, type=int,)
    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for eval.",)
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.",)
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.",)
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.",)
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.",)
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.",)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.",)
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="eval every X updates steps.",)
    parser.add_argument("--seed", type=int, default=2048,
                        help="random seed for initialization",)

    parser.add_argument("--rec_model_dir", default=None, type=str, )
    parser.add_argument("--det_model_dir", default=None, type=str, )
    parser.add_argument(
        "--label_path", default="./labels/labels_ser.txt", type=str, required=False, )
    parser.add_argument(
        "--label_map_path", default="./labels/labels_ser.txt", type=str, required=False, )
    parser.add_argument("--infer_imgs", default=None, type=str, required=False)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ocr_json_path", default=None,
                        type=str, required=False, help="ocr prediction results")
    parser.add_argument("--image_data_dir", default=None, type=str)
    parser.add_argument("--print_step", type=int, default=100)
    # yapf: enable
    args = parser.parse_args()
    return args
