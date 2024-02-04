import itertools
import pathlib

import torch
import torchvision


CIFAR10_PATH = pathlib.Path("/shared/ml/datasets/vision/CIFAR10")
TARGET_PATH = pathlib.Path(__file__).parent / "data_100"
TARGET_FILENAME_TEMPLATE = "class_{}.pth.tar"

if __name__ == '__main__':
    dataset_train = torchvision.datasets.CIFAR10(
        CIFAR10_PATH,
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ],
        ),
    )
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32)

    dataset_test = torchvision.datasets.CIFAR10(
        CIFAR10_PATH,
        download=True,
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ],
        ),
    )
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32)

    classes = {}

    for batch in itertools.chain(iter(dataloader_train), iter(dataloader_test)):
        images, targets = batch

        for index, t in enumerate(targets.flatten()):
            t_py = t.item()
            if t_py not in classes:
                classes[t_py] = []
            # we use [[]] so that the extra dimension for the batch
            # is not lost, hence having shape [1, 3, 32, 32] for CIFAR10
            # and avoiding issues regarding the torch.cat which puts together along
            # the first dimension 
            classes[t_py].append(images[[index]])
        
    classes_final = {t: torch.cat(images) for t, images in classes.items()}

    TARGET_PATH.mkdir(exist_ok=True, parents=True)
    for t, imgs in classes_final.items():
        torch.save(imgs, TARGET_PATH / TARGET_FILENAME_TEMPLATE.format(str(t)))
