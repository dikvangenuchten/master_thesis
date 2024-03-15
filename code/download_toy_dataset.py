import fiftyone


def download_toy():
    fiftyone.config.do_not_track = True
    fiftyone.config.dataset_zoo_dir = "/datasets/fiftyone/"
    fiftyone.config.default_dataset_dir = "/datasets/fiftyone/"

    train = fiftyone.zoo.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["segmentations"],
        classes=["Cat", "Dog"],
        max_samples=100
    )
    train.persistent = True
    test = fiftyone.zoo.load_zoo_dataset(
        "open-images-v7",
        split="test",
        label_types=["segmentations"],
        classes=["Cat", "Dog"],
        max_samples=100
    )
    test.persistent = True
    validation = fiftyone.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["segmentations"],
        classes=["Cat", "Dog"],
        max_samples=100
    )
    validation.persistent = True


if __name__ == "__main__":
    download_toy()
