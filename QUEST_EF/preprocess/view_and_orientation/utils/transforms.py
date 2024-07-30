import albumentations as A
import cv2


def get_transforms(transforms_config):
    augment_apply = transforms_config.get('apply', [])
    img_size = transforms_config['image_size']

    transforms = []

    transforms.append(A.Resize(img_size, img_size))

    if 'rotation' in augment_apply:
        transforms.append(A.SafeRotate(limit=transforms_config.get('rotation_limit', 10), border_mode=cv2.BORDER_CONSTANT, value=0, p=1))

    if 'horizontal_flip' in augment_apply:
        transforms.append(A.HorizontalFlip(p=transforms_config.get('horizontal_filp_p', 0.5)))

    if 'random_crop' in augment_apply:
        transforms.append(A.RandomResizedCrop(img_size, img_size, scale=(transforms_config.get('random_crop_scale', 0.8),0.6)))

    if 'normalize' in augment_apply:
        transforms.append(A.Normalize(0.0998, 0.1759))

    if 'blur' in augment_apply:
        transforms.append(A.AdvancedBlur(p=transforms_config.get('blur_p', 0.5), blur_limit=(3,7)))
        #validation_transforms.append(A.augmentations.transforms.AdvancedBlur(p=transforms_config.get('blur_p', 1)))
    
    if 'brightness' in augment_apply:
        transforms.append(A.RandomBrightnessContrast(p=transforms_config.get('brightness_p', 0.5),brightness_limit=transforms_config.get('brightness_limit', 0.2), contrast_limit=transforms_config.get('contrast_limit', 0.2)))
        #validation_transforms.append(A.augmentations.transforms.RandomBrightnessContrast(p=transforms_config.get('brightness_p', 1)))
    
    if 'sharpen' in augment_apply:
        transforms.append(A.Sharpen(p=transforms_config.get('sharpen_p', 0.5)))
        #validation_transforms.append(A.augmentations.transforms.Sharpen(p=transforms_config.get('sharpen_p', 0.5)))
    
    if 'gamma' in augment_apply:
        transforms.append(A.RandomGamma(p=transforms_config.get('gamma_p', 0.5), gamma_limit=transforms_config.get('gamma_limit', (80,120))))
        #validation_transforms.append(A.augmentations.transforms.RandomGamma(p=transforms_config.get('gamma_p', 0.5)))

    transforms = A.Compose(transforms)

    return transforms
