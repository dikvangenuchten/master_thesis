"""Forked from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/datasets/oxford_pet.py

Modified for compatibility with this repo.
 - Update for PIL>=10.0.0
 - Update for pytorch transforms_v2
 
Dataset README
```
OXFORD-IIIT PET Dataset
-----------------------
Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar

We have created a 37 category pet dataset with roughly 200 images for each class. 
The images have a large variations in scale, pose and lighting. All images have an 
associated ground truth annotation of breed, head ROI, and pixel
level trimap segmentation.

Contents:
--------
trimaps/        Trimap annotations for every image in the dataset
                Pixel Annotations: 1: Foreground 2:Background 3: Not classified
xmls/           Head bounding box annotations in PASCAL VOC Format

list.txt        Combined list of all images in the dataset
                Each entry in the file is of following nature:
                Image CLASS-ID SPECIES BREED ID
                ID: 1:37 Class ids
                SPECIES: 1:Cat 2:Dog
                BREED ID: 1-25:Cat 1:12:Dog
                All images with 1st letter as captial are cat images while
                images with small first letter are dog images.
trainval.txt    Files describing splits used in the paper.However,
test.txt        you are encouraged to try random splits.



Support:
-------
For any queries contact,

Omkar Parkhi: omkar@robots.ox.ac.uk

References:
----------
[1] O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
   Cats and Dogs  
   IEEE Conference on Computer Vision and Pattern Recognition, 2012

Note:
----
Dataset is made available for research purposes only. Use of these images must respect 
the corresponding terms of use of original websites from which they are taken.
See [1] for list of websites.  
```

"""

import os
import shutil
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from torchvision.tv_tensors import Image as TImage
from torchvision.tv_tensors import Mask
from tqdm import tqdm


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}

        self.name = "oxford_pet"
        self.root = os.path.join(root, self.name)
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(
            self.root, "annotations", "trimaps"
        )

        (
            self.filenames,
            self.class_id,
            self.species,
            self.breed_id,
        ) = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        class_id = self.class_id[idx]
        species = self.species[idx]
        breed_id = self.breed_id[idx]

        image_path = os.path.join(
            self.images_directory, filename + ".jpg"
        )
        mask_path = os.path.join(
            self.masks_directory, filename + ".png"
        )

        image = Image.open(image_path).convert("RGB")

        trimap = Image.open(mask_path)

        image, trimap = TImage(image), Mask(trimap, dtype=torch.long)
        if self.transform is not None:
            image, trimap = self.transform(image, trimap)

        return image, trimap, class_id, species, breed_id

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = (
            "test.txt" if self.mode == "test" else "trainval.txt"
        )
        split_filepath = os.path.join(
            self.root, "annotations", split_filename
        )
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames, class_id, species, breed_id = list(
            zip(*[x.split(" ") for x in split_data])
        )
        if self.mode == "train":  # 90% for train
            filenames = [
                x for i, x in enumerate(filenames) if i % 10 != 0
            ]
            class_id = [
                int(x) for i, x in enumerate(class_id) if i % 10 != 0
            ]
            species = [
                int(x) for i, x in enumerate(species) if i % 10 != 0
            ]
            breed_id = [
                int(x) for i, x in enumerate(breed_id) if i % 10 != 0
            ]
        elif self.mode == "valid":  # 10% for validation
            filenames = [
                x for i, x in enumerate(filenames) if i % 10 == 0
            ]
            class_id = [
                int(x) for i, x in enumerate(class_id) if i % 10 == 0
            ]
            species = [
                int(x) for i, x in enumerate(species) if i % 10 == 0
            ]
            breed_id = [
                int(x) for i, x in enumerate(breed_id) if i % 10 == 0
            ]
        return filenames, class_id, species, breed_id

    @staticmethod
    def download(root):
        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class OxfordPetForegroundDataset(OxfordPetDataset):
    @property
    def class_map(self) -> dict:
        return {0: "background", 1: "foreground"}

    def __getitem__(self, idx):
        (
            image,
            trimap,
            _class_id,
            _species,
            _breed_id,
        ) = super().__getitem__(idx)
        mask = self._preprocess_tensor_mask(trimap)
        return image, mask

    @staticmethod
    def _preprocess_tensor_mask(mask):
        mask[mask == 2] = 0
        mask[(mask == 1) | (mask == 3)] = 1
        return mask


class OxfordSpeciesDataset(OxfordPetDataset):
    @property
    def class_map(self) -> dict:
        return {0: "background", 1: "cat", 2: "dog"}

    def __getitem__(self, idx):
        (
            image,
            trimap,
            _class_id,
            species,
            _breed_id,
        ) = super().__getitem__(idx)
        mask = self._preprocess_tensor_mask(trimap)
        return image, mask * species

    @staticmethod
    def _preprocess_tensor_mask(mask):
        mask[mask == 2] = 0
        mask[(mask == 1) | (mask == 3)] = 1
        return mask


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(
            url, filename=filepath, reporthook=t.update_to, data=None
        )
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)
