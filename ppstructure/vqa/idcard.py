import json
import os
import cv2
import paddle
import copy
from paddle.io import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from augment import augment
from constants import ALL_EXCLUDE_FIELD

__all__ = ['IDCardDataset']

def normalize_bbox(bbox, width, height):
    return [
                min(1000, max(0, int(1000 * (bbox[0] / width)))),
                min(1000, max(0, int(1000 * (bbox[1] / height)))),
                min(1000, max(0, int(1000 * (bbox[2] / width)))),
                min(1000, max(0, int(1000 * (bbox[3] / height)))),
            ]

class IDCardDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 image_data_dir,
                 label_path,
                 contains_re=False,
                 label2id_map=None,
                 entities_label2id_map=None,
                 img_size=(224, 224),
                 pad_token_label_id=None,
                 add_special_ids=False,
                 return_attention_mask=True,
                 load_mode='all',
                 max_seq_len=512,
                 online_augment=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_data_dir = image_data_dir
        self.label_path = label_path
        self.contains_re = contains_re
        self.label2id_map = label2id_map
        self.img_size = img_size
        self.pad_token_label_id = pad_token_label_id
        self.add_special_ids = add_special_ids
        self.return_attention_mask = return_attention_mask
        self.load_mode = load_mode
        self.max_seq_len = max_seq_len
        self.online_augment = online_augment

        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

        self.all_lines = self.read_all_lines()
        random.shuffle(self.all_lines)

        self.entities_labels = entities_label2id_map
        self.return_keys = {
            'bbox': {
                'type': 'np',
                'dtype': 'int64'
            },
            'input_ids': {
                'type': 'np',
                'dtype': 'int64'
            },
            'labels': {
                'type': 'np',
                'dtype': 'int64'
            },
            'attention_mask': {
                'type': 'np',
                'dtype': 'int64'
            },
            'image': {
                'type': 'np',
                'dtype': 'float32'
            },
            'token_type_ids': {
                'type': 'np',
                'dtype': 'int64'
            },
            'entities': {
                'type': 'dict'
            },
            'relations': {
                'type': 'dict'
            }
        }

        if load_mode == "all":
            self.encoded_inputs_all = self._parse_label_file_all()

    def pad_sentences(self,
                      encoded_inputs,
                      max_seq_len=512,
                      pad_to_max_seq_len=True,
                      return_attention_mask=True,
                      return_token_type_ids=True,
                      truncation_strategy="longest_first",
                      return_overflowing_tokens=False,
                      return_special_tokens_mask=False):
        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
            max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            if self.tokenizer.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.tokenizer.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.tokenizer.pad_token_id] * difference
                encoded_inputs["labels"] = encoded_inputs[
                    "labels"] + [self.pad_token_label_id] * difference
                encoded_inputs["bbox"] = encoded_inputs[
                    "bbox"] + [[0, 0, 0, 0]] * difference
            elif self.tokenizer.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.tokenizer.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
                encoded_inputs["labels"] = [
                    self.pad_token_label_id
                ] * difference + encoded_inputs["labels"]
                encoded_inputs["bbox"] = [
                    [0, 0, 0, 0]
                ] * difference + encoded_inputs["bbox"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        return encoded_inputs

    def truncate_inputs(self, encoded_inputs, max_seq_len=512):
        for key in encoded_inputs:
            if key == "sample_id":
                continue
            length = min(len(encoded_inputs[key]), max_seq_len)
            encoded_inputs[key] = encoded_inputs[key][:length]
        return encoded_inputs

    def read_all_lines(self, ):
        with open(self.label_path, "r", encoding='utf-8') as fin:
            lines = fin.readlines()
        return lines

    def _parse_label_file_all(self):
        """
        parse all samples
        """
        encoded_inputs_all = []
        for line in self.all_lines:
            encoded_inputs_all.extend(self._parse_label_file(line))
        return encoded_inputs_all

    def _parse_label_file(self, line):
        """
        parse single sample
        """

        image_name, info_str = line.split("\t")
        card_type = '_'.join(image_name.split('_')[:-1])
        image_path = os.path.join(self.image_data_dir, card_type, image_name)
        # self.visualize(image_path, info_str)
        def add_imgge_path(x):
            x['image_path'] = image_path
            return x

        encoded_inputs = self._read_encoded_inputs_sample(info_str, image_path)
        if self.contains_re:
            encoded_inputs = self._chunk_re(encoded_inputs)
        else:
            encoded_inputs = self._chunk_ser(encoded_inputs)
        encoded_inputs = list(map(add_imgge_path, encoded_inputs))
        return encoded_inputs

    def _get_label(self, label, image_path):
        if label == 'loai_xe' and \
            ('gdk_back' in image_path \
            or 'dk_oto_back' in image_path \
            or 'dk_xemay_back' in image_path):
            label = 'loai_xe'
        elif label == 'key_noi_cap' and 'cmca_front' in image_path:
            label = 'other'
        elif label.isdigit() \
            or label.startswith('en_') \
                or label.startswith('unit_') \
                    or label.startswith('tires_') \
                        or label.endswith('_en') \
                            or label.endswith('_ta') \
                                or label.startswith('loai_xe') \
                                    or (label.startswith('hang') and label != 'hang_cap') \
                                        or label in ALL_EXCLUDE_FIELD:
            label = 'other'

        return label

    def _read_encoded_inputs_sample(self, info_str, image_path):
        """
        parse label info
        """
        # read text info
        info_dict = json.loads(info_str)
        height = info_dict["height"]
        width = info_dict["width"]

        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        gt_label_list = []

        if self.contains_re:
            # for re
            entities = []
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()
        
        ocr_info = info_dict['ocr_info']
        if self.online_augment: # Random remove textlines
            ocr_info = augment.random_remove_textlines(ocr_info)

        for info in ocr_info:
            text = info["text"]
            label = info["label"]
            if (label == 'van_tay_trai' or label == 'van_tay_phai') and 'cm_back' in image_path:
                continue
            if self.contains_re:
                # for re
                if len(info["text"]) == 0:
                    empty_entity.add(info["id"])
                    continue
                id2label[info["id"]] = info["label"]
                relations.extend([tuple(sorted(l)) for l in info["linking"]])

            # x1, y1, x2, y2
            bbox = info["bbox"]

            # Normalize bbox
            bbox = normalize_bbox(bbox, height, width)

            if self.online_augment: 
                # Text augment
                text = augment.augment_text(text)

            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True)
            # encoded_res['input_ids'] = [[CLS], ..., [SEP]] 
            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res["input_ids"] = encode_res["input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:
                                                                            -1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:
                                                                            -1]
            gt_label = []
            keys = list(self.entities_labels.keys())
            label = self._get_label(label, image_path)
            if label == 'other' or label not in keys:
                gt_label.extend([self.label2id_map['O']] * len(encode_res["input_ids"]))
            else:
                gt_label.append(self.label2id_map[("B-" + label)])
                gt_label.extend([self.label2id_map[("I-" + label)]] *
                                (len(encode_res["input_ids"]) - 1))

            if self.contains_re:
                if gt_label[0] != self.label2id_map["O"]:
                    entity_id_to_index_map[info["id"]] = len(entities)
                    entities.append({
                        "start": len(input_ids_list),
                        "end":
                        len(input_ids_list) + len(encode_res["input_ids"]),
                        "label": label.upper(),
                    })
            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))
            gt_label_list.extend(gt_label)

        encoded_inputs = {
            "input_ids": input_ids_list,
            "labels": gt_label_list,
            "token_type_ids": token_type_ids_list,
            "bbox": bbox_list,
            "attention_mask": [1] * len(input_ids_list),
        }
        encoded_inputs = self.pad_sentences(
            encoded_inputs,
            max_seq_len=self.max_seq_len,
            return_attention_mask=self.return_attention_mask)
        encoded_inputs = self.truncate_inputs(encoded_inputs)

        if self.contains_re:
            relations = self._relations(entities, relations, id2label,
                                        empty_entity, entity_id_to_index_map)
            encoded_inputs['relations'] = relations
            encoded_inputs['entities'] = entities
        return encoded_inputs

    def _chunk_ser(self, encoded_inputs):
        encoded_inputs_all = []
        seq_len = len(encoded_inputs['input_ids'])
        chunk_size = 512
        for chunk_id, index in enumerate(range(0, seq_len, chunk_size)):
            chunk_beg = index
            chunk_end = min(index + chunk_size, seq_len)
            encoded_inputs_example = {}
            for key in encoded_inputs:
                encoded_inputs_example[key] = encoded_inputs[key][chunk_beg:
                                                                  chunk_end]

            encoded_inputs_all.append(encoded_inputs_example)
        return encoded_inputs_all

    def _chunk_re(self, encoded_inputs):
        # prepare data
        entities = encoded_inputs.pop('entities')
        relations = encoded_inputs.pop('relations')
        encoded_inputs_all = []
        chunk_size = 512
        for chunk_id, index in enumerate(
                range(0, len(encoded_inputs["input_ids"]), chunk_size)):
            item = {}
            for k in encoded_inputs:
                item[k] = encoded_inputs[k][index:index + chunk_size]

            # select entity in current chunk
            entities_in_this_span = []
            global_to_local_map = {}  #
            for entity_id, entity in enumerate(entities):
                if (index <= entity["start"] < index + chunk_size and
                        index <= entity["end"] < index + chunk_size):
                    entity["start"] = entity["start"] - index
                    entity["end"] = entity["end"] - index
                    global_to_local_map[entity_id] = len(entities_in_this_span)
                    entities_in_this_span.append(entity)

            # select relations in current chunk
            relations_in_this_span = []
            for relation in relations:
                if (index <= relation["start_index"] < index + chunk_size and
                        index <= relation["end_index"] < index + chunk_size):
                    relations_in_this_span.append({
                        "head": global_to_local_map[relation["head"]],
                        "tail": global_to_local_map[relation["tail"]],
                        "start_index": relation["start_index"] - index,
                        "end_index": relation["end_index"] - index,
                    })
            item.update({
                "entities": reformat(entities_in_this_span),
                "relations": reformat(relations_in_this_span),
            })
            item['entities']['label'] = [
                self.entities_labels[x] for x in item['entities']['label']
            ]
            encoded_inputs_all.append(item)
        return encoded_inputs_all

    def _relations(self, entities, relations, id2label, empty_entity,
                   entity_id_to_index_map):
        """
        build relations
        """
        relations = list(set(relations))
        relations = [
            rel for rel in relations
            if rel[0] not in empty_entity and rel[1] not in empty_entity
        ]
        kv_relations = []
        for rel in relations:
            pair = [id2label[rel[0]], id2label[rel[1]]]
            if 'key' in rel[0]:
                kv_relations.append({
                    "head": entity_id_to_index_map[rel[0]],
                    "tail": entity_id_to_index_map[rel[1]]
                })
            elif 'key' in rel[1]:
                kv_relations.append({
                    "head": entity_id_to_index_map[rel[1]],
                    "tail": entity_id_to_index_map[rel[0]]
                })
            else:
                continue
        relations = sorted(
            [{
                "head": rel["head"],
                "tail": rel["tail"],
                "start_index": get_relation_span(rel, entities)[0],
                "end_index": get_relation_span(rel, entities)[1],
            } for rel in kv_relations],
            key=lambda x: x["head"], )
        return relations

    def load_img(self, image_path):
        # read img
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize_h, resize_w = self.img_size
        im_shape = img.shape[0:2]
        im_scale_y = resize_h / im_shape[0]
        im_scale_x = resize_w / im_shape[1]
        img_new = cv2.resize(
            img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)
        mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
        img_new = img_new / 255.0
        img_new -= mean
        img_new /= std
        img = img_new.transpose((2, 0, 1))
        return img

    def __getitem__(self, idx):
        if self.load_mode == "all":
            data = copy.deepcopy(self.encoded_inputs_all[idx])
        else:
            data = self._parse_label_file(self.all_lines[idx])[0]

        image_path = data.pop('image_path')
        data["image"] = self.load_img(image_path)

        return_data = {}
        for k, v in data.items():
            if k in self.return_keys:
                if self.return_keys[k]['type'] == 'np':
                    v = np.array(v, dtype=self.return_keys[k]['dtype'])
                return_data[k] = v
        return return_data

    def __len__(self, ):
        if self.load_mode == "all":
            return len(self.encoded_inputs_all)
        else:
            return len(self.all_lines)


    def visualize(self, image_path, info_str):
        if 'pp' not in image_path:
            return 
        color = 'red'
        def draw_box_txt(draw, bbox, label, font):
            draw.rectangle(bbox, fill=color)

            size = font.getsize(label)
            start_y = max(0, bbox[1] - size[1])
            draw.rectangle([bbox[0], start_y, bbox[0] + size[0], start_y + size[1]], fill=color)
            draw.text((bbox[0], start_y), label, fill='black', font=font)

        font = ImageFont.load_default()
        image = Image.open(image_path)
        img_new = image.copy()
        draw = ImageDraw.Draw(img_new)
        for item in json.loads(info_str)['ocr_info']:
            bbox = item['bbox']
            label = item['label']
            if label != 'key_ngay_cap':
                continue
            label = self._get_label(label, image_path)
            draw_box_txt(draw, bbox, label, font)
        img_new = Image.blend(image, img_new, 0.3)
        img_new.save(os.path.join('./debug', os.path.basename(image_path)))

def get_relation_span(rel, entities):
    bound = []
    for entity_index in [rel["head"], rel["tail"]]:
        bound.append(entities[entity_index]["start"])
        bound.append(entities[entity_index]["end"])
    return min(bound), max(bound)


def reformat(data):
    new_data = {}
    for item in data:
        for k, v in item.items():
            if k not in new_data:
                new_data[k] = []
            new_data[k].append(v)
    return new_data
