from typing import Callable, Literal, Dict, Optional

import torch
import torchvision
from torchvision.tv_tensors import Image, Mask
import fiftyone
import tqdm


fiftyone.config.do_not_track = True
OUTPUT_TYPES = Literal["img", "latent", "semantic_mask"]


class FiftyOneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "validation"] = "train",
        transform: Optional[Callable]=None,
        output_structure: Dict[str, OUTPUT_TYPES] = {
            "input": "latent",
            "target": "semantic_mask",
        },
        max_samples: int = 1000
    ):
        fiftyone.config.dataset_zoo_dir = "/datasets/fiftyone/"
        fiftyone.config.default_dataset_dir = "/datasets/fiftyone/"
        
        self.transform = transform
        
        self.classes = ["Cat", "Dog"]
        
        self._inner_dataset = fiftyone.zoo.load_zoo_dataset(
            "open-images-v7",
            split=split,
            classes=self.classes,
            label_types=["segmentations"],
            max_samples=max_samples,
            persistent=True
        )
        self._ids = [x.id for x in tqdm.tqdm(self._inner_dataset)]
        
        self._full_class_map = self._inner_dataset.info["classes_map"]
        
        self.classes.insert(0, "Unlabeled")
        self._cat_id_to_semantic = {
            c: i for i, c in enumerate(self.classes)
        }
        self._cat_id_to_semantic["Unlabeled"] = 0
        
        self.class_map = {i: c for i, c in enumerate(self.classes)}
        self.class_map[0] = "Unlabeled"

        # self._inner_dataset.persistent = True
        self.output_structure = output_structure
        
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        item = self._inner_dataset[self._ids[index]]
        image = Image(torchvision.io.read_image(item["filepath"]))
        mask = torch.zeros_like(image[0], dtype=torch.long)
        _, h, w = image.shape
        for detection in item.ground_truth.detections:
            # TODO add instance to the output
            target = 1
            instance_mask = detection.to_segmentation(frame_size=(w, h), target=target).mask
            mask[instance_mask == 1] = self._cat_id_to_semantic.get(detection.label, 0)

        mask = Mask(mask)
        out = {"input": image, "target": mask, "image": image}

        if self.transform is not None:
            return self.transform(out)
        return out
    
    def _get_type(self, index, type_) -> torch.Tensor:
        if type_ == "img":
            return ...
        
