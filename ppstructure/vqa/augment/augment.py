import numpy as np
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug as ia
from PIL import Image, ImageDraw
from .custom_augmentation import (LightFlare, ParallelLight,
                                 SpotLight, WarpTexture, 
                                 RandomLine, Blob, Shadow)
import re
import random
from augment.text_augment import TextAugmentation
text_augmentation = TextAugmentation()
prop = {
    "remove_accents": 0.005,
    'random_change_character': 0.03,
    'random_remove_space': 0.0001,
    'random_add_space': 0.002,
    'random_remove_word': 0.003,
}

def random_remove_textlines(ocr_info):
    '''
    Randomly remove textlines
    '''
    ret = []
    for info in ocr_info:
        key = info['label']
        if 'title' in key or 'cnxh' in key: #Random remove 5%
            if random.uniform(0, 1) > 0.05:
                ret.append(info)
        elif 'key' in key: #Random remove 3%
            if random.uniform(0, 1) > 0.03:
                ret.append(info)
        else: # Random remove 2%
            if random.uniform(0, 1) > 0.02:
                ret.append(info)

    return ret

def augment_text(words):
    augmented_words = []
    for w in words:
        splitted = re.split('_| ', text_augmentation.augment([w], prop=prop)[0].strip())
        augmented_w = ' '.join(splitted)
        augmented_words.append(augmented_w)

    return augmented_words

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def normalize_box(bbox, width, height):
    return [
                min(1000, max(0, int(1000 * (bbox[0] / width)))),
                min(1000, max(0, int(1000 * (bbox[1] / height)))),
                min(1000, max(0, int(1000 * (bbox[2] / width)))),
                min(1000, max(0, int(1000 * (bbox[3] / height)))),
            ]

def augment_image(image, boxes):
    image = np.array(image)
    height, width, _ = image.shape
    polygon_lst = []
    for box in boxes:
        unnormalized_box = unnormalize_box(box, image.shape[1], image.shape[0])
        polygon_lst.append(Polygon([[unnormalized_box[0], unnormalized_box[1]], 
                                    [unnormalized_box[2], unnormalized_box[1]], 
                                    [unnormalized_box[2], unnormalized_box[3]], 
                                    [unnormalized_box[0], unnormalized_box[3]]]))
    psoi = PolygonsOnImage(polygon_lst, shape=image.shape)

    augmented_img, augmented_boxes = augment_pipeline_1(
        images=[image], polygons=[psoi]
    )

    augmented_img = Image.fromarray(augmented_img[0])
    augmented_boxes = augmented_boxes[0]
    normal_boxes = []
    # draw = ImageDraw.Draw(augmented_img)
    for augmented_box in augmented_boxes.polygons:
        polygon = augmented_box.exterior
        x_min = int(min(polygon[:, 0]))
        y_min = int(min(polygon[:, 1]))
        x_max = int(max(polygon[:, 0]))
        y_max = int(max(polygon[:, 1]))
        box = [x_min, y_min, x_max, y_max]
        box = normalize_box(box, width, height)
        normal_boxes.append(box)
        # draw.polygon(polygon, outline='red')
    # augmented_img.save('1.jpg')
    return augmented_img, normal_boxes

def blur_augment():
    aug = iaa.OneOf([
            iaa.GaussianBlur(sigma=(1.0, 3.0)),
            iaa.AverageBlur(k=(2, 3)),
            iaa.MedianBlur(k=3),
            iaa.MotionBlur(k=(3, 5)),
            iaa.BilateralBlur(d=(3, 4), sigma_color=(10, 250), sigma_space=(10, 250)),

            iaa.imgcorruptlike.DefocusBlur(severity=1),
            iaa.imgcorruptlike.GlassBlur(severity=1),
            
            iaa.imgcorruptlike.Pixelate(severity=(1,3)),
    ])
    return aug

def noise_augment():
    aug = iaa.OneOf([
            iaa.Pepper(0.1),
            iaa.AdditiveGaussianNoise(scale=(0, 0.01*255), per_channel=True),
            iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=True),
            iaa.AdditivePoissonNoise(40, per_channel=True),
            iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
            iaa.Dropout(p=(0, 0.2), per_channel=0.5),
            iaa.ReplaceElementwise(0.1, iap.Normal(128, 0.4*128), per_channel=0.5),
            iaa.imgcorruptlike.SpeckleNoise(severity=1),
    ])
    return aug 

def weather_augment():
    aug = iaa.OneOf([
            iaa.imgcorruptlike.Brightness(severity=(1,3)),
            iaa.imgcorruptlike.Saturate(severity=1),
            iaa.pillike.EnhanceSharpness(),
            iaa.imgcorruptlike.Spatter(severity=(1,3)),
            iaa.CoarseDropout(0.02, size_percent=0.01, per_channel=1),
            iaa.imgcorruptlike.Contrast(severity=1),
            iaa.imgcorruptlike.Snow(severity=1),
            iaa.imgcorruptlike.Frost(severity=1),
            iaa.imgcorruptlike.Fog(severity=(1, 3)),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
            iaa.MultiplyBrightness((0.8, 1.05)),
            iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((-20, 20)))),
            iaa.ChangeColorTemperature((2000, 40000))
    ])
    return aug

def blend_augment():
    aug = iaa.OneOf([
        # Blend list
        iaa.BlendAlphaVerticalLinearGradient(
            # list_augmenter,
            iaa.Clouds(),
            start_at=(0.0, 1.0), end_at=(0.0, 1.0)
        ),
        
        iaa.BlendAlphaFrequencyNoise(
            exponent=(-4,4),
            foreground=(
                # list_augmenter
                iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
            ),
            size_px_max=32,
            upscale_method="linear",
            iterations=1,
            sigmoid=False
        ),

        iaa.BlendAlphaMask(
            iaa.InvertMaskGen(0.35, iaa.VerticalLinearGradientMaskGen()),
            iaa.Clouds()
        ),
        iaa.BlendAlphaFrequencyNoise(
            exponent=(-4,4),
            foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
            size_px_max=32,
            upscale_method="linear",
            iterations=1,
            sigmoid=False,
            per_channel=True
        ),
        iaa.SimplexNoiseAlpha(
            iaa.Multiply(iap.Choice([0.8, 1.2]), per_channel=True)
        ),
        
    ])
    return aug

augment_pipeline_1 = iaa.Sequential([
    iaa.Sometimes(0.05, LightFlare()),
    iaa.Sometimes(0.05, ParallelLight()),
    iaa.Sometimes(0.1, SpotLight()),
    iaa.Sometimes(0.05, RandomLine()),
    iaa.Sometimes(0.05, Blob()),
    iaa.Sometimes(0.05, WarpTexture()),
    iaa.Sometimes(0.05, Shadow()),
    iaa.Sometimes(0.3, 
        iaa.OneOf([
            iaa.Affine(
                scale = {"x": (1, 1.25), "y": (1, 1.25)},
                rotate=(-30, 30),
                shear=(-10, 10),
                translate_percent=(-0.2, 0.2),
                cval=(0,255),
                mode='constant',
                fit_output=True
            ),
            iaa.PerspectiveTransform(scale=(0.01, 0.05), mode=ia.ALL ,keep_size=True, 
                            fit_output=True, polygon_recoverer="auto", cval=(0,255)),
        ])
    ),

    iaa.Sometimes(0.1, iaa.Sequential([
        blur_augment(),
        noise_augment(),
        blend_augment(),
        weather_augment()
    ])),
])