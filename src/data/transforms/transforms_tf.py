import numpy as np
from PIL import Image
from random import uniform

class Transform(object):
    ''' data augmentationに用いる画像を変換する関数を定義するためのクラス。
    '''

    def __init__(self,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 scale_var=0.2):  # origin 0.2, tmp1.5
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.scale_var = scale_var
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def resize(self, img, size, resample=None):
        img = Image.fromarray(img)
        size = size[::-1]
        if resample == 'nearest':
            resized_img = img.resize(size=size, resample=Image.NEAREST)
        else:
            resized_img = img.resize(size=size, resample=Image.BILINEAR)
        return np.array(resized_img)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y = y[:, ::-1]
        return img, y

    def normalize(self, img):
        normalized_img = np.array(img / 127.5 - 1, dtype=np.float32)
        return normalized_img

    def crop(self, array, position):
        y, x, height, width = position
        return array[y:y + height, x:x + width]

    def random_crop(self, img, gt, cropped_size):
        img_height, img_width = img.shape[:-1]
        crop_height, crop_width = cropped_size

        xlim = img_width - crop_width
        ylim = img_height - crop_height

        x = int(uniform(0, xlim))
        y = int(uniform(0, ylim))
        cropped_img = self.crop(img, (y, x, crop_height, crop_width))
        cropped_gt = self.crop(gt, (y, x, crop_height, crop_width))
        return cropped_img, cropped_gt

    def scale(self, img, y):
        # alpha = self.scale_var * np.random.random() + 0.5 #tmp
        alpha = self.scale_var * np.random.random() + 1
        # height = int(alpha * img.shape[0]) #tmp
        # width = int(alpha * img.shape[1]) #tmp
        height = int(1 / alpha * img.shape[0])
        width = int(1 / alpha * img.shape[1])
        cropped_img, cropped_y = self.random_crop(img, y, (height, width))
        scaled_img = self.resize(cropped_img, img.shape[:-1], Image.BILINEAR)
        scaled_y = self.resize(cropped_y, y.shape, Image.NEAREST)
        return scaled_img, scaled_y

    def random_transform(self, img, y):
        '''
        論文再現のため、一時OFF
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            img = jitter(img)
        if self.lighting_std:
            img = self.lighting(img)
        '''
        if self.hflip_prob > 0:
            img, y = self.horizontal_flip(img, y)
        img = img.astype(np.uint8)
        if self.scale_var:
            img, y = self.scale(img, y)
        y = y.astype(np.uint8)
        return img, y