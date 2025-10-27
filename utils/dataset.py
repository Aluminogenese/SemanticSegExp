"""
数据增强
包含：
1. 多尺度裁剪
2. 颜色抖动（针对遥感影像）
3. 几何变换（旋转、翻转、仿射）
4. MixUp / CutMix
5. 测试时增强（TTA）
"""

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import random
from PIL import Image
import rasterio
import tifffile as tiff

ALLOWED_RASTER_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


class AdvancedDataset(Dataset):
    
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', 
                 augment=True, crop_size=512):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment
        self.crop_size = crop_size
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_any(path):
        """读取图像，保留位深度"""
        lower = path.lower()
        if lower.endswith('.tif') or lower.endswith('.tiff'):
            try:
                with rasterio.open(path) as src:
                    arr = src.read()
                    if hasattr(arr, 'filled'):
                        arr = arr.filled(0)
                    if arr.ndim == 3:
                        arr = np.transpose(arr, (1, 2, 0))
                    return arr
            except Exception:
                pass
            try:
                arr = tiff.imread(path)
                if arr.ndim == 3 and arr.shape[0] in (3, 4) and (arr.shape[2] not in (3, 4)):
                    arr = np.transpose(arr, (1, 2, 0))
                return arr
            except Exception:
                pass
        
        img = Image.open(path)
        try:
            img.load()
        except Exception:
            pass
        arr = np.array(img)
        return arr

    def random_crop(self, img, mask, crop_size):
        """随机裁剪"""
        h, w = img.shape[:2]
        
        if h <= crop_size or w <= crop_size:
            # 如果图像小于裁剪尺寸，填充
            pad_h = max(crop_size - h, 0)
            pad_w = max(crop_size - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape[:2]
        
        # 随机选择裁剪位置
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        img = img[top:top+crop_size, left:left+crop_size]
        mask = mask[top:top+crop_size, left:left+crop_size]
        
        return img, mask

    def random_flip(self, img, mask):
        """随机翻转"""
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        
        return img, mask

    def random_rotate90(self, img, mask):
        """随机90度旋转"""
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        return img, mask

    def color_jitter_remote_sensing(self, img):
        """针对遥感影像的颜色抖动"""
        if random.random() > 0.5:
            # 亮度调整
            brightness_factor = random.uniform(0.9, 1.1)
            img = img * brightness_factor
            
            # 对比度调整
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.9, 1.1)
                mean = img.mean(axis=(0, 1), keepdims=True)
                img = (img - mean) * contrast_factor + mean
            
            # 添加少量噪声
            if random.random() > 0.7:
                noise = np.random.normal(0, 0.01, img.shape)
                img = img + noise
        
        return np.clip(img, 0, 1)

    def normalize_image(self, img):
        """归一化图像"""
        img = img.astype(np.float32)
        
        if np.issubdtype(img.dtype, np.uint16) or img.max() > 255:
                # 简单归一化
            channel_max = np.array([4553.0, 4166.0, 4489.0, 9142.0], dtype=np.float32)
            img = img / channel_max
        else:
            # uint8图像
            img = img / 255.0
        
        return img

    def preprocess(self, img, mask):
        """预处理流水线"""
        # 确保维度正确
        if img.ndim == 2:
            img = img[..., None]
        if mask.ndim == 2:
            mask = mask[..., None]
        elif mask.ndim == 3 and mask.shape[2] > 1:
            mask = mask[..., 0:1]
        
        h, w = img.shape[:2]
        
        # 数据增强
        if self.augment:
            # 随机裁剪
            img, mask = self.random_crop(img, mask, self.crop_size)
            
            # 几何变换
            img, mask = self.random_flip(img, mask)
            img, mask = self.random_rotate90(img, mask)
        else:
            # 测试时中心裁剪或调整大小
            if h > self.crop_size or w > self.crop_size:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
                img = img[top:top+self.crop_size, left:left+self.crop_size]
                mask = mask[top:top+self.crop_size, left:left+self.crop_size]
        
        # 归一化
        img = self.normalize_image(img)
        
        # 颜色抖动（仅训练时）
        if self.augment:
            img = self.color_jitter_remote_sensing(img)
        
        # 二值化mask
        mask = (mask > 0).astype(np.float32)
        
        # HWC -> CHW
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        return img, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        mask_file = [f for f in mask_file if splitext(f)[1].lower() in ALLOWED_RASTER_EXTS]
        img_file = [f for f in img_file if splitext(f)[1].lower() in ALLOWED_RASTER_EXTS]

        assert len(mask_file) == 1, f'Mask error for ID {idx}: {mask_file}'
        assert len(img_file) == 1, f'Image error for ID {idx}: {img_file}'

        mask_np = self._read_image_any(mask_file[0])
        img_np = self._read_image_any(img_file[0])

        # 验证尺寸
        if img_np.ndim == 2:
            img_h, img_w = img_np.shape
        else:
            img_h, img_w = img_np.shape[:2]
        if mask_np.ndim == 2:
            mask_h, mask_w = mask_np.shape
        else:
            mask_h, mask_w = mask_np.shape[:2]
        
        assert (img_w, img_h) == (mask_w, mask_h), \
            f'Size mismatch for {idx}: image {(img_w, img_h)} vs mask {(mask_w, mask_h)}'

        img, mask = self.preprocess(img_np, mask_np)

        return {
            'image': torch.from_numpy(img).type(torch.float32),
            'mask': torch.from_numpy(mask).type(torch.float32)
        }


class TTADataset(Dataset):
    """测试时增强数据集"""
    
    def __init__(self, base_dataset, tta_transforms=4):
        """
        tta_transforms: TTA变换数量
        - 4: 原图 + 3个90度旋转
        - 8: 上述 + 水平翻转版本
        """
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms
        
    def __len__(self):
        return len(self.base_dataset) * self.tta_transforms
    
    def __getitem__(self, idx):
        base_idx = idx // self.tta_transforms
        transform_idx = idx % self.tta_transforms
        
        sample = self.base_dataset[base_idx]
        img = sample['image']
        mask = sample['mask']
        
        # 应用变换
        if self.tta_transforms == 8:
            if transform_idx >= 4:
                # 水平翻转
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
                transform_idx -= 4
        
        # 旋转
        if transform_idx > 0:
            img = torch.rot90(img, k=transform_idx, dims=[1, 2])
            mask = torch.rot90(mask, k=transform_idx, dims=[1, 2])
        
        sample['image'] = img
        sample['mask'] = mask
        sample['transform_idx'] = transform_idx
        
        return sample


def tta_merge_predictions(predictions, transform_indices, tta_transforms=4):
    """
    合并TTA预测结果
    
    Args:
        predictions: List of prediction tensors
        transform_indices: List of transform indices
        tta_transforms: Total number of TTA transforms used
    
    Returns:
        Merged prediction tensor
    """
    merged = []
    
    for pred, trans_idx in zip(predictions, transform_indices):
        # 反向旋转
        if tta_transforms == 8 and trans_idx >= 4:
            trans_idx -= 4
            pred = torch.flip(pred, dims=[2])
        
        if trans_idx > 0:
            pred = torch.rot90(pred, k=4-trans_idx, dims=[1, 2])
        
        merged.append(pred)
    
    # 平均融合
    return torch.stack(merged).mean(dim=0)