import random
import string
import pandas as pb
from unidecode import unidecode
from multiprocessing import Pool
import copy
import numpy as np

class TextAugmentation:
    def __init__(self):
        # self.p = Pool(4)

        MAPPING_CHARACTER_TO_NUMBER = {
            'a': ['a', 'ạ', 'u'],
            'à': ['à', 'a', 'ă', 'â', 'ạ'],
            'á': ['á', 'a', 'ă', 'â', 'ạ'],
            'ả': ['ả', 'a', 'ă', 'â', 'à', 'á', 'ạ'],
            'ạ': ['ạ', 'a'],
            'ã': ['ả', 'a', 'ă', 'â', 'à', 'á', 'ạ'],

            'â': ['â', 'ả', 'a', 'ã', 'ô', 'ạ'],
            'ầ': ['ầ', 'â', 'ả', 'a', 'ồ', 'ô', 'ạ'],
            'ấ': ['ấ', 'â', 'ẫ', 'ả', 'a', 'ồ', 'ô', 'ạ'],
            'ẩ': ['ẩ', 'â', 'ẫ', 'ả', 'a', 'ổ', 'ạ'],
            'ậ': ['ậ', 'â', 'ẫ', 'ả', 'a', 'ỗ', 'ô', 'ạ'],
            'ẫ': ['ẫ', 'â', 'a', 'ỗ', 'ô', 'ạ'],

            'ă': ['ă', 'ã', 'ả', 'a', 'ạ'],
            'ằ': ['ằ', 'ă', 'ã', 'à', 'a', 'ạ'],
            'ắ': ['ắ', 'ă', 'ã', 'á', 'a', 'ạ'],
            'ẳ': ['ẳ', 'ẵ', 'ả', 'a', 'ạ'],
            'ặ': ['ặ', 'ă', 'ã', 'a', 'ạ'],
            'ẵ': ['ẵ', 'ă', 'ã', 'a', 'ạ'],

            'b': ['b', 'h', 'lo', 'io', '|o'],
            'c': ['c', 'C', '(', '<'],
            'd': ['d', 'đ', 'ol', 'o|'],
            'đ': ['đ', 'd', 'ol', 'o|'],

            'e': ['e', 'ẹ', 'c'],

            'è': ['è', 'e', 'ẹ', 'ê', 'c'],
            'é': ['é', 'e', 'ẹ', 'ê', 'c'],
            'ẻ': ['ẻ', 'e', 'ẹ', 'ê', 'ẽ', 'c'],
            'ẹ': ['ẹ', 'e', 'c'],
            'ẽ': ['ẽ', 'e', 'ẹ', 'ê', 'c'],

            'ê': ['ê', 'e', 'ẽ', 'è', 'é', 'ẹ'],
            'ề': ['ề', 'ê', 'e', 'ẽ', 'è', 'é', 'ẹ'],
            'ế': ['ế', 'ê', 'e', 'ẽ', 'é', 'é', 'ẹ'],
            'ể': ['ể', 'ê', 'e', 'ẽ', 'ẻ', 'é', 'ẹ'],
            'ệ': ['ệ', 'ê', 'e', 'ẽ', 'ẹ', 'é', 'ẹ'],
            'ễ': ['ễ', 'ê', 'e', 'ẽ', 'ẽ', 'é', 'ẹ'],

            'f': ['f', 't', 'l', 'I', 'i', 'ì', 'í', 'ỉ', 'ĩ', '1', ],
            'g': ['g', 'y', 'q',],
            'h': ['h', 'b', 'ln', '|n'],

            'i': ['i', '1', 'f', 'j', 'l', 't', 'ì', 'í', 'ị', 'ỉ', 'ĩ'],
            'ì': ['ì', 'i', '1', 'f', 'j', 'l', 't', 'ì', 'í', 'ị', 'ỉ', 'ĩ'],
            'í': ['í', 'i', '1', 'f', 'j', 'l', 't', 'ì', 'í', 'ị', 'ỉ', 'ĩ'],
            'ỉ': ['ỉ', 'i', '1', 'f', 'j', 'l', 't', 'ì', 'í', 'ị', 'ỉ', 'ĩ'],
            'ị': ['ị', 'i', '1', 'f', 'j', 'l', 't', 'ì', 'í', 'ị', 'ỉ', 'ĩ'],
            'ĩ': ['ĩ', 'i', '1', 'f', 'j', 'l', 't', 'ì', 'í', 'ị', 'ỉ', 'ĩ'],
            'j': ['j', 'i', 'J', '1', 'f', 'l', 't',  'I',  'ì', 'í', 'ị', 'ỉ'],
            'k': ['k', '|<'],
            'l': ['l', 't', 'j', 'i', '1', 'I', 'ì', 'í', 'ị', 'ỉ'],
            'm': ['m', '111', '11i', '1i1', '1ii', 'i11', 'i1i', 'ii1', 'iii', '1n',
                  'n1', 'in', 'ni', 'rn', 'rii', 'r11', '1r1', 'rr1', 'rll',
                  'lll', '1ll', 'll1', 'l1l', 'lr1'],
            'n': ['n', '11', 'ii', '1i', 'i1', 'll', 'l1', 'lr', '1l', '11', 'tt'],

            'o': ['o', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],
            'ò': ['ò', 'Ò', 'ô', 'o', 'O', '0', '()', '<>', 'ọ', 'Ọ'],
            'ó': ['ó', 'Ó', 'ô', 'o', 'O', '0', 'c', '()', '<>', 'ọ', 'Ọ'],
            'ỏ': ['ỏ', 'Ỏ', 'ô', 'o', 'O', '0', 'c', '()', '<>', 'ọ', 'Ọ'],
            'ọ': ['ọ', 'Ọ', 'o', 'O', '0', 'c', '()', '<>', 'ọ', 'Ọ'],
            'õ': ['õ', 'Õ', 'ô', 'o', 'O', '0', 'c', '()', '<>', 'ọ', 'Ọ'],

            'ô': ['ô', 'õ', 'Ô', 'o', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],
            'ồ': ['ồ', 'o', 'ô', 'õ', 'Ô', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],
            'ố': ['ố', 'o', 'ô', 'õ', 'Ô', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],
            'ổ': ['ổ', 'o', 'ô', 'õ', 'Ô', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],
            'ộ': ['ộ', 'o', 'ô', 'õ', 'Ô', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],
            'ỗ': ['ỗ', 'o', 'ô', 'õ', 'Ô', 'O', '0', '()', 'c', '<>', 'ọ', 'Ọ'],

            'ơ': ['ơ', 'ỏ', 'o', 'Ơ', 'ọ'],
            'ờ': ['ờ', 'ơ', 'ỏ', 'o', 'Ờ', 'ọ'],
            'ớ': ['ớ', 'ơ', 'ỏ', 'o', 'Ớ', 'ọ'],
            'ở': ['ở', 'ơ', 'ỏ', 'o', 'Ở', 'ọ'],
            'ợ': ['ợ', 'ơ', 'ỏ', 'o', 'Ợ', 'ọ'],
            'ỡ': ['ỡ', 'ơ', 'õ', 'o', 'Ỡ', 'ọ'],

            'p': ['p', '|o'],
            'q': ['q', 'g', 'y', 'o|'],
            'r': ['r', 'í', 'f'],
            's': ['s', 'S', '5'],
            't': ['t', 'i', 'l'],

            '0': ['0', 'O', 'o', '()', '<>', 'Ọ'],
            '1': ['1', 'I', 'l', 't', 'f'],
            '2': ['2', 'Z'],
            '3': ['3'],
            '4': ['4'],
            '5': ['5', 'S', 's'],
            '6': ['6'],
            '7': ['7'],
            '8': ['8', 'B'],
            '9': ['9'],

            'u': ['u', 'ụ'],
            'ù': ['ù', 'u', 'ụ'],
            'ú': ['ú', 'u', 'ụ'],
            'ủ': ['ủ', 'ư', 'u', 'ụ'],
            'ụ': ['ụ', 'u'],
            'ũ': ['ũ', 'u', 'ụ'],

            'ư': ['ư', 'ủ', 'u', 'ụ'],
            'ừ': ['ừ', 'ư', 'u', 'ụ'],
            'ứ': ['ứ', 'ư', 'u', 'ụ'],
            'ử': ['ử', 'ư', 'u', 'ụ'],
            'ự': ['ự', 'ư', 'u', 'ụ'],
            'ữ': ['ữ', 'ư', 'u', 'ụ'],

            'v': ['v', 'V', '\/'],
            'w': ['w', 'vv', 'uu', 'vu', 'uv', 'W', 'VV', 'UU', 'UV', 'VU'],
            'x': ['x', 'X'],
            'y': ['y', 'g'],
            'z': ['z', '2', 'Z'],

            'A': ['A', 'Ạ'],
            'À': ['À', 'Â', 'A', 'Ạ'],
            'Á': ['Á', 'Â', 'A', 'Ạ'],
            'Ả': ['Ả', 'Â', 'A', 'Ạ'],
            'Ạ': ['Ạ', 'A', 'Ạ'],
            'Ã': ['Ã', 'Â', 'Ã', 'A', 'Ạ'],

            'Â': ['Â', 'Ả', 'A', 'Ạ'],
            'Ầ': ['Ầ', 'Â', 'Ả', 'A', 'Ạ'],
            'Ấ': ['Ấ', 'Â', 'Ả', 'A', 'Ạ'],
            'Ẩ': ['Ẩ', 'Â', 'Ả', 'A', 'Ạ'],
            'Ậ': ['Ậ', 'Â', 'Ả', 'A', 'Ạ'],
            'Ẫ': ['Ẫ', 'Â', 'Ả', 'A', 'Ạ'],

            'Ă': ['Ã', 'Ả', 'A', 'Ạ'],
            'Ằ': ['À', 'Ã', 'Ả', 'A', 'Ạ'],
            'Ắ': ['Á', 'Ã', 'Ả', 'A', 'Ạ'],
            'Ẳ': ['Ẳ', 'Ã', 'Ả', 'A', 'Ạ'],
            'Ặ': ['Ặ', 'Ã', 'Ả', 'A', 'Ạ'],
            'Ẵ': ['Ẵ', 'Ã', 'Ả', 'A', 'Ạ'],

            'B': ['B', '8'],
            'C': ['C', 'c', '(', '<'],
            'D': ['D', 'Đ', 'O', '0'],
            'Đ': ['Đ', 'D', 'Đ', 'O', '0'],

            'E': ['E', 'Ẹ'],
            'È': ['È', 'Ê', 'E', 'Ẹ'],
            'É': ['É', 'Ê', 'E', 'Ẹ'],
            'Ẻ': ['Ẻ', 'Ê', 'E', 'Ẹ'],
            'Ẹ': ['Ẹ', 'E', 'Ẹ'],
            'Ẽ': ['Ẽ', 'Ê', 'E', 'Ẹ'],

            'Ê': ['Ê', 'Ẻ', 'Ẽ', 'E', 'Ẹ'],
            'Ề': ['Ề', 'Ê', 'Ẻ', 'Ẽ', 'E', 'Ẹ'],
            'Ế': ['Ế', 'Ê', 'Ẻ', 'Ẽ', 'E', 'Ẹ'],
            'Ể': ['Ể', 'Ê', 'Ẻ', 'Ẽ', 'E', 'Ẹ'],
            'Ệ': ['Ệ', 'Ê', 'Ẻ', 'Ẽ', 'E', 'Ẹ'],
            'Ễ': ['Ễ', 'Ê', 'Ẻ', 'Ẽ', 'E', 'Ẹ'],

            'F': ['F', 'P'],
            'G': ['G', '6'],
            'H': ['H', '11', 'II'],

            'I': ['I', 'Ị', 'L', '|'],
            'Ì': ['I', 'Ị', 'L', '|'],
            'Í': ['I', 'Ị', 'L', '|'],
            'Ỉ': ['I', 'Ị', 'L', '|'],
            'Ị': ['I', 'Ị', 'L', '|'],
            'Ĩ': ['I', 'Ị', 'L', '|'],

            'J': ['J', 'i'],
            'K': ['K', '|<'],
            'L': ['L', '|_', 'I', '|'],
            'M': ['M'],
            'N': ['N'],

            'O': ['O', '0', 'o', '()', '<>', 'Ọ'],
            'Ò': ['Ò', 'ò', 'Ô', 'O', '0', 'o', '()', '<>', 'Ọ'],
            'Ó': ['Ó', 'ó', 'Ô', 'O', '0', 'o', '()', '<>', 'Ọ'],
            'Ỏ': ['Ỏ', 'ỏ', 'Ô', 'O', '0', 'o', '()', '<>', 'Ọ'],
            'Ọ': ['Ọ', 'ọ', 'O', '0', 'o', '()', '<>', 'Ọ'],
            'Õ': ['Õ', 'Ô', 'O', '0', 'o', '()', '<>', 'Ọ'],

            'Ô': ['Ô', 'Ỏ', 'Õ', 'ô', 'O', '0'],
            'Ồ': ['Ồ', 'Ô', 'Ỏ', 'Õ', 'ô', 'O', '0'],
            'Ố': ['Ố', 'Ô', 'Ỏ', 'Õ', 'ô', 'O', '0'],
            'Ổ': ['Ổ', 'Ô', 'Ỏ', 'Õ', 'ô', 'O', '0'],
            'Ộ': ['Ô', 'Ô', 'Ỏ', 'Õ', 'ô', 'O', '0'],
            'Ỗ': ['Ỗ', 'Ô', 'Ỏ', 'Õ', 'ô', 'O', '0'],

            'Ơ': ['Ơ', 'Ỏ', 'ơ', 'O', '0'],
            'Ờ': ['Ờ', 'Ơ', 'Ỏ', 'ơ', 'O', '0'],
            'Ớ': ['Ớ', 'Ơ', 'Ỏ', 'ơ', 'O', '0'],
            'Ở': ['Ở', 'Ơ', 'Ỏ', 'ơ', 'O', '0'],
            'Ỡ': ['Ỡ', 'Ơ', 'Ỏ', 'ơ', 'O', '0'],
            'Ợ': ['Ợ', 'Ơ', 'Ỏ', 'ơ', 'O', '0'],

            'P': ['P', 'F'],
            'Q': ['Q', 'G'],
            'R': ['R'],
            'S': ['S', 's', '5'],
            'T': ['T'],

            'U': ['U', 'Ụ'],
            'Ù': ['Ù', 'U', 'Ụ'],
            'Ú': ['Ú', 'U', 'Ụ'],
            'Ủ': ['Ủ', 'Ư', 'U', 'Ụ'],
            'Ụ': ['Ụ', 'U', 'Ụ'],
            'Ũ': ['Ũ', 'Ư', 'U', 'Ụ'],

            'Ư': ['Ư', 'Ủ', 'U', 'Ụ'],
            'Ừ': ['Ừ', 'Ư', 'Ủ', 'U', 'Ụ'],
            'Ứ': ['Ứ', 'Ư', 'Ủ', 'U', 'Ụ'],
            'Ử': ['Ử', 'Ư', 'Ủ', 'U', 'Ụ'],
            'Ự': ['Ự', 'Ư', 'Ủ', 'U', 'Ụ'],
            'Ữ': ['Ữ', 'Ư', 'Ủ', 'U', 'Ụ'],

            'V': ['V', 'v', '\/'],
            'W': ['W', 'VV', 'w', 'UU', 'VU', 'UV', 'W'],
            'X': ['X', 'x'],
            'Y': ['Y'],
            'Z': ['Z', '2', 'z']
        }
        MAPPING_CHARACTER_TO_NUMBER_SET = {
            x: set(MAPPING_CHARACTER_TO_NUMBER[x]) for x in
            MAPPING_CHARACTER_TO_NUMBER}

        MAPPING_CHARACTER_TO_NUMBER_TEMP = copy.deepcopy(
            MAPPING_CHARACTER_TO_NUMBER)
        for x in MAPPING_CHARACTER_TO_NUMBER:
            for y in MAPPING_CHARACTER_TO_NUMBER_SET[x]:
                if y not in MAPPING_CHARACTER_TO_NUMBER_TEMP and (len(y) > 1 or (not unidecode(y).isalpha() and not unidecode(y).isnumeric())):
                    MAPPING_CHARACTER_TO_NUMBER_TEMP[y] = []
                    MAPPING_CHARACTER_TO_NUMBER_SET[y] = set({})
                else:
                    continue
                if x not in MAPPING_CHARACTER_TO_NUMBER_SET[y]:
                    MAPPING_CHARACTER_TO_NUMBER_TEMP[y].append(x)
                    MAPPING_CHARACTER_TO_NUMBER_SET[y].add(x)

        MAX_LEN = 100
        for x in MAPPING_CHARACTER_TO_NUMBER_TEMP:
            while len(MAPPING_CHARACTER_TO_NUMBER_TEMP[x]) < MAX_LEN:
                MAPPING_CHARACTER_TO_NUMBER_TEMP[x].append(x)
        self.MAPPING_CHARACTER_TO_NUMBER = MAPPING_CHARACTER_TO_NUMBER_TEMP

        TEMP_PROP = [1/(i+1) for i in range(MAX_LEN)]
        self.PROP = TEMP_PROP / np.sum(TEMP_PROP)

    def remove_accents(self, texts: [], propability=0.05):
        def _remove_accents(text: str):
            return unidecode(text)

        return [_remove_accents(x) for x in texts]

    def random_change_character(self, texts: [], propability=0.05):
        def _change_char(x):
            return np.random.choice(self.MAPPING_CHARACTER_TO_NUMBER[x], p=self.PROP)

        def _random_change_character(text):
            new_text = ""

            i = 0
            while i < len(text):
               if i + 2 < len(text) and text[i:i+3] in self.MAPPING_CHARACTER_TO_NUMBER and random.uniform(0,1) < propability:
                  new_text += _change_char(text[i:i+3])
                  i += 3
               elif i + 1 < len(text) and text[i:i+2] in self.MAPPING_CHARACTER_TO_NUMBER and random.uniform(0,1) < propability:
                  new_text += _change_char(text[i:i + 2])
                  i += 2
               elif text[i] in self.MAPPING_CHARACTER_TO_NUMBER and random.uniform(0,1) < propability:
                  new_text += _change_char(text[i])
                  i += 1
               else:
                  new_text += text[i]
                  i += 1

            return new_text

        return [_random_change_character(x) for x in texts]

    def random_remove_space(self, texts: [], propability=0.05):
        def _random_remove_space(text):
            return "".join([x for x in text if not (
                        x == " " and random.uniform(0, 1) < propability)])

        return [_random_remove_space(x) for x in texts]

    def random_add_space(self, texts: [], propability=0.05):
        def _random_add_space(text):
            return "".join(
                [x + " " if random.uniform(0, 1) < propability else x for x in
                 text])

        return [_random_add_space(x) for x in texts]

    def random_remove_word(self, texts: [], propability=0.05):
        def _random_remove_word(text):
            return "".join([x for x in text if random.uniform(0, 1) > propability])
        return [_random_remove_word(x) for x in texts]

    def augment(self, texts: [], prop: {}):
        if "remove_accents" in prop and prop[
            'remove_accents'] > 0 and random.uniform(0, 1) < prop[
            'remove_accents']:
            texts = self.remove_accents(texts)

        if "random_remove_space" in prop and prop['random_remove_space'] > 0:
            texts = self.random_remove_space(texts,
                                             prop['random_remove_space'])

        if "random_add_space" in prop and prop['random_add_space'] > 0:
            texts = self.random_add_space(texts, prop['random_add_space'])

        if "random_change_character" in prop and prop['random_change_character'] > 0:
            texts = self.random_change_character(texts, prop[
                'random_change_character'])

        if "random_remove_word" in prop and prop['random_remove_word']:
            texts = self.random_remove_word(texts, prop['random_remove_word'])

        return texts


if __name__ == '__main__':
    model = TextAugmentation()

    input = [u"Nguyễn Hồng Sơn là con lợn", "Ta là ai đây là đâu?",
             "Ta có 04 con gà!"]

    prop = {
    "remove_accents": 0,
    'random_change_character': 0.03,
    'random_remove_space': 0.00,
    'random_add_space': 0.002,
    'random_remove_word': 0.003,
}
    outputs = model.augment(texts=input, prop=prop)

    print(outputs)

    all_characters = "".join([x for x in model.MAPPING_CHARACTER_TO_NUMBER])
