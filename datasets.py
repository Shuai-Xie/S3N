import os
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from nest import register


@register
def image_transform(
        image_size: Union[int, List[int]],
        augmentation: dict,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    horizontal_flip = augmentation.pop('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1

    vertical_flip = augmentation.pop('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1

    random_crop = augmentation.pop('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)

    center_crop = augmentation.pop('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))

    if len(augmentation) > 0:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))

    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0], **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]

    return transforms.Compose([v for v in t if v is not None])


@register
def fetch_data(
        dataset: Callable[[str], Dataset],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_splits: List[str] = [],
        test_splits: List[str] = [],
        train_shuffle: bool = True,
        test_shuffle: bool = False,
        test_image_size: int = 600,
        train_augmentation: dict = {},
        test_augmentation: dict = {},
        batch_size: int = 1,
        test_batch_size: Optional[int] = None) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """
    :param dataset: Callable, `fgvc_datasets.fgvc_dataset`
            [str] data_dir: ./datasets/CUB_200_2011
            Dataset: `fgvc_datasets.FGVC_Dataset`
    :param transform: train transform
    :param target_transform: label transform
    :param num_workers:
    :param pin_memory:
    :param drop_last:
    :param train_splits:
    :param test_splits:
    :param train_shuffle:
    :param test_shuffle:
    :param test_image_size:
    :param train_augmentation:
    :param test_augmentation:
    :param batch_size:
    :param test_batch_size:
    :return:
        train_loader_list, test_loader_list: [ ('train': DataLoader object) ,... ]
    """

    train_transform = transform(augmentation=train_augmentation) if transform else None
    # train_transfrom use transform own image_size: [448, 448]
    train_loader_list = []
    for split in train_splits:  # ['train']
        train_loader_list.append(
            (split, DataLoader(  # torch.dataloader
                # dataset has set `data_dir` in yml
                dataset=dataset(split=split,
                                transform=train_transform,
                                target_transform=target_transform),
                batch_size=batch_size,  # 16
                num_workers=num_workers,  # 4
                pin_memory=pin_memory,  # True
                drop_last=drop_last,  # False
                shuffle=train_shuffle  # True
            )))

    test_transform = transform(image_size=[test_image_size, test_image_size],
                               augmentation=test_augmentation) if transform else None
    test_loader_list = []
    for split in test_splits:  # ['test']
        test_loader_list.append(
            (split, DataLoader(
                dataset=dataset(split=split,
                                transform=test_transform,
                                target_transform=target_transform),
                batch_size=batch_size if test_batch_size is None else test_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                # above use cfgs as train
                shuffle=test_shuffle
            )))

    return train_loader_list, test_loader_list
