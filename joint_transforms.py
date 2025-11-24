import random

from PIL import Image
from PIL.Image import frombytes
from torchvision import transforms
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Compose3(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, edge):
        assert img.size == mask.size == edge.size, f"Image sizes do not match: {img.size}, {mask.size}, {edge.size}"
        for t in self.transforms:
            img, mask, edge = t(img, mask, edge)
        return img, mask, edge

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask,edge):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT)

        # elif random.random() < 0.6:
        #     return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)

        else:
            return img, mask, edge

class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask,edge):
        assert img.size == mask.size and img.size == edge.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST),edge.resize(self.size, Image.NEAREST)

class RandomHorizontallyFlip3(object):
    def __call__(self, img, mask, edge):

        if random.random() < 0.3:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT),edge.transpose(Image.FLIP_LEFT_RIGHT)
        elif random.random() < 0.6:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), edge.transpose(
                Image.FLIP_TOP_BOTTOM)
        else:
            return img, mask, edge

class Resize3(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask, edge):
        assert img.size == mask.size and img.size == edge.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.BILINEAR)



class RandomResizedCrop_transpose(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label, edge):
        if random.random() < 0.5:
            h, w = img.size
            #print(img.size)
            x1 = random.randint(0, max(0, w - self.ch))
            y1 = random.randint(0, max(0, h - self.cw))
            img_crop = img.crop((y1, x1, y1 + self.cw, self.ch + x1))
            label_crop = label.crop((y1, x1, y1 + self.cw, self.ch + x1))
            edge_crop = label.crop((y1, x1, y1 + self.cw, self.ch + x1))

            #img_crop = img_crop.resize((w, h))
            #label_crop = label_crop.resize((w, h))
            return img_crop, label_crop, edge_crop
        else:
            return img, label, edge

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask, edge):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), edge.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return img, mask, edge

class RandomRotate(object):
    def __init__(self, degrees=10, p=0.3):
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (-degrees, degrees)
        self.p = p
    def __call__(self, img, mask, edge):
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            img = img.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
            edge = edge.rotate(angle, resample=Image.NEAREST)
            return img, mask, edge
        else:
            return img, mask, edge

class RandomResizedCrop3(object):
    def __init__(self, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5):
        self.scale = scale
        self.ratio = ratio
        self.p = p
    def get_params(self, width, height):
        area = width * height
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            log_ratio = (random.uniform(self.ratio[0], self.ratio[1]))
            w = int(round((target_area * log_ratio) ** 0.5))
            h = int(round((target_area / log_ratio) ** 0.5))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return j, i, w, h
        w = min(width, height)
        h = w
        i = (height - h) // 2
        j = (width - w) // 2
        return j, i, w, h
    def __call__(self, img, mask, edge):
        if random.random() < self.p:
            width, height = img.size
            j, i, w, h = self.get_params(width, height)
            img = img.crop((j, i, j + w, i + h))
            mask = mask.crop((j, i, j + w, i + h))
            edge = edge.crop((j, i, j + w, i + h))
            return img, mask, edge
        else:
            return img, mask, edge
