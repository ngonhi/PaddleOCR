from imgaug import augmenters as iaa
from scipy.stats import norm
import random
import cv2
import numpy as np 
import glob
import os
import copy
import pylab
import scipy.ndimage as ndi
import blend_modes
from .shadow_augmenter import shadow


class ParallelLight(iaa.meta.Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _decayed_value_in_norm(self, x, max_value, min_value, center, range):
        """
        decay from max value to min value following Gaussian/Normal distribution
        """
        radius = range / 3
        center_prob = norm.pdf(center, center, radius)
        x_prob = norm.pdf(x, center, radius)
        x_value = (x_prob / center_prob) * (max_value - min_value) + min_value
        return x_value


    def _decayed_value_in_linear(self, x, max_value, padding_center, decay_rate):
        """
        decay from max value to min value with static linear decay rate.
        """
        x_value = max_value - abs(padding_center - x) * decay_rate
        if x_value < 0:
            x_value = 1
        return x_value


    def generate_parallel_light_mask(self, mask_size,
                                     position=None,
                                     direction=None,
                                     max_brightness=255,
                                     min_brightness=0,
                                     mode="gaussian",
                                     linear_decay_rate=None):
        """
        Generate decayed light mask generated by light strip given its position, direction
        Args:
            mask_size: tuple of integers (w, h) defining generated mask size
            position: tuple of integers (x, y) defining the center of light strip position,
                    which is the reference point during rotating
            direction: integer from 0 to 360 to indicate the rotation degree of light strip
            max_brightness: integer that max brightness in the mask
            min_brightness: integer that min brightness in the mask
            mode: the way that brightness decay from max to min: linear or gaussian
            linear_decay_rate: only valid in linear_static mode. Suggested value is within [0.2, 2]
        Return:
            light_mask: ndarray in float type consisting value from 0 to strength
        """
        if position is None:
            pos_x = random.randint(0, mask_size[0])
            pos_y = random.randint(0, mask_size[1])
        else:
            pos_x = position[0]
            pos_y = position[1]
        if direction is None:
            direction = random.randint(0, 360)
            # print("Rotate degree: ", direction)
        if linear_decay_rate is None:
            if mode == "linear_static":
                linear_decay_rate = random.uniform(0.2, 2)
            if mode == "linear_dynamic":
                linear_decay_rate = (
                    max_brightness - min_brightness) / max(mask_size)
        assert mode in ["linear_dynamic", "linear_static", "gaussian"], \
            "mode must be linear_dynamic, linear_static or gaussian"
        padding = int(max(mask_size) * np.sqrt(2))
        # add padding to satisfy cropping after rotating
        canvas_x = padding * 2 + mask_size[0]
        canvas_y = padding * 2 + mask_size[1]
        mask = np.zeros(shape=(canvas_y, canvas_x), dtype=np.float32)
        # initial mask's up left corner and bottom right corner coordinate
        init_mask_ul = (int(padding), int(padding))
        init_mask_br = (int(padding+mask_size[0]), int(padding+mask_size[1]))
        init_light_pos = (padding + pos_x, padding + pos_y)
        # fill in mask row by row with value decayed from center
        for i in range(canvas_y):
            if mode == "linear":
                i_value = self._decayed_value_in_linear(
                    i, max_brightness, init_light_pos[1], linear_decay_rate)
            elif mode == "gaussian":
                i_value = self._decayed_value_in_norm(
                    i, max_brightness, min_brightness, init_light_pos[1], mask_size[1])
            else:
                i_value = 0
            mask[i] = i_value
        # rotate mask
        rotate_M = cv2.getRotationMatrix2D(init_light_pos, direction, 1)
        mask = cv2.warpAffine(mask, rotate_M, (canvas_x,  canvas_y))
        # crop
        mask = mask[init_mask_ul[1]:init_mask_br[1],
                    init_mask_ul[0]:init_mask_br[0]]
        mask = np.asarray(mask, dtype=np.uint8)
        # add median blur
        mask = cv2.medianBlur(mask, 9)
        mask = 255 - mask
        return mask

    def add_parallel_light(self, image, light_position=None, direction=None, max_brightness=255, min_brightness=0,
                           mode="gaussian", linear_decay_rate=None, transparency=None):
        """
        Add mask generated from parallel light to given image
        """
        if transparency is None:
            transparency = random.uniform(0.5, 0.85)
        height, width, _ = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = self.generate_parallel_light_mask(mask_size=(width, height),
                                                 position=light_position,
                                                 direction=direction,
                                                 max_brightness=max_brightness,
                                                 min_brightness=min_brightness,
                                                 mode=mode,
                                                 linear_decay_rate=linear_decay_rate)
        hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image[image > 255] = 255
        image = np.asarray(image, dtype=np.uint8)
        return image

    def augment_image(self, image, **kwargs):
        return self.add_parallel_light(image)

    def _augment_images(self, images, **kwargs):
        return [self.add_parallel_light(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return


class SpotLight(iaa.meta.Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

    def generate_spot_light_mask(self, mask_size,
                                 position=None,
                                 max_brightness=255,
                                 min_brightness=0,
                                 mode="gaussian",
                                 linear_decay_rate=None,
                                 speedup=False):
        """
        Generate decayed light mask generated by spot light given position, direction. Multiple spotlights are accepted.
        Args:
            mask_size: tuple of integers (w, h) defining generated mask size
            position: list of tuple of integers (x, y) defining the center of spotlight light position,
                    which is the reference point during rotating
            max_brightness: integer that max brightness in the mask
            min_brightness: integer that min brightness in the mask
            mode: the way that brightness decay from max to min: linear or gaussian
            linear_decay_rate: only valid in linear_static mode. Suggested value is within [0.2, 2]
            speedup: use `shrinkage then expansion` strategy to speed up vale calculation
        Return:
            light_mask: ndarray in float type consisting value from max_brightness to min_brightness. If in 'linear' mode
                        minimum value could be smaller than given min_brightness.
        """
        if position is None:
            position = [(random.randint(0, mask_size[0]),
                         random.randint(0, mask_size[1]))]
        if linear_decay_rate is None:
            if mode == "linear_static":
                linear_decay_rate = random.uniform(0.25, 1)
        assert mode in ["linear", "gaussian"], \
            "mode must be linear_dynamic, linear_static or gaussian"
        mask = np.zeros(shape=(mask_size[1], mask_size[0]), dtype=np.float32)
        if mode == "gaussian":
            mu = np.sqrt(mask.shape[0]**2+mask.shape[1]**2)
            dev = mu / 3.5
            mask = self._decay_value_radically_norm_in_matrix(
                mask_size, position, max_brightness, min_brightness, dev)
        mask = np.asarray(mask, dtype=np.uint8)
        # add median blur
        mask = cv2.medianBlur(mask, 5)
        mask = 255 - mask
        return mask

    def _decay_value_radically_norm_in_matrix(self, mask_size, centers, max_value, min_value, dev):
        """
        _decay_value_radically_norm function in matrix format
        """
        center_prob = norm.pdf(0, 0, dev)
        x_value_rate = np.zeros((mask_size[1], mask_size[0]))
        for center in centers:
            coord_x = np.arange(mask_size[0])
            coord_y = np.arange(mask_size[1])
            xv, yv = np.meshgrid(coord_x, coord_y)
            dist_x = xv - center[0]
            dist_y = yv - center[1]
            dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
            x_value_rate += norm.pdf(dist, 0, dev) / center_prob
        mask = x_value_rate * (max_value - min_value) + min_value
        mask[mask > 255] = 255
        return mask

    def _decay_value_radically_norm(self, x, centers, max_value, min_value, dev):
        """
        Calculate point value decayed from center following Gaussian decay. If multiple centers are given, value
        from each center sums up while limiting the accumulated value into [0, 255]
        NOTE: assuming light at each center is identical: same brightness and same decay rate
        """
        center_prob = norm.pdf(0, 0, dev)
        x_value_rate = 0
        for center in centers:
            distance = np.sqrt((center[0]-x[0])**2 + (center[1]-x[1])**2)
            x_value_rate += norm.pdf(distance, 0, dev) / center_prob
        x_value = x_value_rate * (max_value - min_value) + min_value
        x_value = 255 if x_value > 255 else x_value
        return x_value

    def add_spot_light(self, image, light_position=None, max_brightness=255, min_brightness=0,
                       mode='gaussian', linear_decay_rate=None, transparency=None):
        """
        Add mask generated from spot light to given image
        """
        if transparency is None:
            transparency = random.uniform(0.5, 0.85)
        height, width, _ = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = self.generate_spot_light_mask(mask_size=(width, height),
                                             position=light_position,
                                             max_brightness=max_brightness,
                                             min_brightness=min_brightness,
                                             mode=mode,
                                             linear_decay_rate=linear_decay_rate)
        hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        image[image > 255] = 255
        image = np.asarray(image, dtype=np.uint8)
        return image

    def augment_image(self, image, **kwargs):
        return self.add_spot_light(image)

    def _augment_images(self, images, **kwargs):
        return [self.add_spot_light(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return

class LightFlare(iaa.meta.Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

    def flare_source(self, image, point, radius, src_color):
        overlay = image.copy()
        output = image.copy()
        num_times = radius//10
        alpha = np.linspace(0.0, 1, num=num_times)
        rad = np.linspace(1, radius, num=num_times)
        for i in range(num_times):
            cv2.circle(overlay, point, int(rad[i]), src_color, -1)
            alp = alpha[num_times-i-1] * \
                alpha[num_times-i-1]*alpha[num_times-i-1]
            cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)
        return output


    def augment_image(self, image, hooks=None):
        radius = int(image.shape[1]*random.uniform(0.05, 0.2))
        x = random.randint(0, image.shape[1])
        y = random.randint(0, image.shape[0])
        color = (random.randint(230, 255), random.randint(
            230, 255), random.randint(230, 255))
        return self.flare_source(image, (x, y), radius, color)


    def _augment_images(self, images, **kwargs):
        return [self.augment_image(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return


class WarpTexture(iaa.meta.Augmenter):
    def __init__(self, texture_path='/mnt/ssd/marley/OCR/LayoutXLM/augment/texture', name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.texture_file = glob.glob(os.path.join(texture_path, '*.jpg')) + glob.glob(os.path.join(texture_path, '*.png'))

    def adjust_alpha_channel(self, image, alpha_value=255):
        ''' Adjust alpha channel of image
        '''

        image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        image_mask = cv2.split(image_rgba)[-1]
        image_mask_inv = cv2.bitwise_not(image_mask) 
        if random.random() < 0.10:
            image_mask_inv = alpha_value
        else:  
            is_horizontal = True if random.random() < 0.5 else False
            alpha_gradient = self.get_gradient_2d(random.randrange(alpha_value, 255), random.randrange(alpha_value, 255), image.shape[1], image.shape[0], is_horizontal)
            image_mask_inv = alpha_gradient 
        image_rgba[:,:, -1] = image_mask_inv

        image = cv2.bitwise_and(image_rgba, image_rgba, mask = image_mask)
        return image 


    def get_gradient_2d(self, start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def augment_image(self, image):
        hei, wid = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        texture_path = random.choice(self.texture_file)
        texture_image = cv2.imread(texture_path)
        rotation = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        texture_image = cv2.rotate(texture_image, random.choice(rotation))
        
        if texture_image.shape[0] > image.shape[0] and texture_image.shape[1] > image.shape[1]:
            max_x = abs(texture_image.shape[1] - wid)
            max_y = abs(texture_image.shape[0] - hei)
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            crop_texture = texture_image[y: y + hei, x: x + wid]
            texture_image = cv2.resize(crop_texture, (wid, hei))
        else: 
            texture_image = cv2.resize(texture_image, (wid, hei))
        
        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2BGRA)

        alpha = random.randint(0, 254)
        # alpha = 255
        texture_image = self.adjust_alpha_channel(texture_image, alpha)

        foreground = texture_image.astype(np.float32)
        background = image.astype(np.float32)

        ratio = random.random()
        opacity = random.uniform(0.5, 0.9)
        
        if ratio < 0.5:
            warped_image = blend_modes.lighten_only(background, foreground, opacity)
        else:
            warped_image = blend_modes.soft_light(background, foreground, opacity)

        warped_image = warped_image.astype(np.uint8)
        return warped_image[:,:,:3]

    def _augment_images(self, images, **kwargs):
        return [self.augment_image(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return


def random_blobs(shape, blobdensity, size, roughness=2.0):
    h, w = shape[0], shape[1]
    numblobs = int(blobdensity * w * h)
    mask = np.zeros((h, w), 'i')
    for i in range(numblobs):
        mask[random.randint(0, h-1), random.randint(0, w-1)] = 1
    dt = ndi.distance_transform_edt(1-mask)
    mask = np.array(dt < size, 'f')
    mask = ndi.gaussian_filter(mask, size/(2*roughness))
    mask -= np.amin(mask)
    mask /= np.amax(mask)
    noise = pylab.rand(h, w)
    noise = ndi.gaussian_filter(noise, size/(2*roughness))
    noise -= np.amin(noise)
    noise /= np.amax(noise)
    return np.array(mask * noise > 0.5, 'int')

def augment_blobs(image):
    bg = random_blobs(image.shape, 2e-4, size=10 + random.randint(-3, 10))
    redImg = copy.deepcopy(image)
    if random.random() < 0.5:
        color_blob = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
    else: 
        color_blob = (random.randint(12, 70), random.randint(12, 70), random.randint(12, 70))

    redImg[bg == 1] = color_blob

    alpha = random.uniform(0.3, 0.8)
    cv2.addWeighted(redImg, alpha, image, 1-alpha, 0, image)

    return image

class Blob(iaa.meta.Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

    def augment_image(self, image, hooks=None):
        image = augment_blobs(image)
        return image
    
    def _augment_images(self, images, **kwargs):
        return [self.augment_image(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return


class Shadow(iaa.meta.Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

    def augment_image(self, image, hooks=None):
        image = shadow.add_n_random_shadows(image)
        return image
    
    def _augment_images(self, images, **kwargs):
        return [self.augment_image(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return


class RandomLine(iaa.meta.Augmenter):
    def __init__(self, is_binary=False, name=None, deterministic=False, random_state=None):
        self.middleline_thickness = [1, 2]
        self.middleline_thickness_p = [0.8, 0.2]
        self.is_binary = is_binary
        super().__init__(name=name, deterministic=deterministic, random_state=random_state)

    def augment_image(self, image, **kwargs):
        h, w, = image.shape[:2]
        for _ in range(random.randint(0, 5)):
            thickness = np.random.choice(
                self.middleline_thickness, p=self.middleline_thickness_p)
            image = cv2.line(image,
                             (random.randint(0, w), random.randint(0, h)),
                             (random.randint(0, w), random.randint(0, h)),
                             color=(random.randint(0, 255), random.randint(
                                 0, 255), random.randint(0, 255)) if not self.is_binary else random.choice([(0, 0, 0), (255, 255, 255)]),
                             thickness=thickness,
                             lineType=cv2.LINE_AA)
        return image

    def _augment_images(self, images, **kwargs):
        return [self.augment_image(each_img) for each_img in images]

    def _augment_keypoints(self, **kwargs):
        return

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return