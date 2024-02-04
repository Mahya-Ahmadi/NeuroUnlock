import functools
import operator
import os
import pathlib
import random
import typing

import numpy
import torch
import torchvision

CIFAR10_BATCH_SIZE = 32
CIFAR10_INPUT_FEATURES = 3072
CIFAR10_TRANSFORM = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
CIFAR10_DATASET_CLASS = torchvision.datasets.CIFAR10
CIFAR10_PATH = '/shared/ml/datasets/vision/CIFAR10/'
DEVICE = torch.device("cuda")
EPOCHS = 30
LEARNING_RATE = 0.001
LOSS_FN = torch.nn.CrossEntropyLoss()
MODEL_CHECKPOINT_PATH = pathlib.Path(__file__).parent / "checkpoint"
MODEL_CHECKPOINT_BASENAME = "model.pt"
OPTIMIZER_CLASS = torch.optim.Adam
SEED = 41
TRAINING_LOG_FILE = MODEL_CHECKPOINT_PATH / "model_training.log"
VALIDATION_LOG_FILE = MODEL_CHECKPOINT_PATH / "model_validation.log"


class BalancedSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, n_classes, length, balanced=True):
        length_per_class = length // n_classes
        subset_length = 0
        classes = {}
        for index in range(len(dataset)):
            inp, target = dataset[index]
            index_list = classes.setdefault(target, [])
            if balanced and len(index_list) >= length_per_class:
                continue
            elif subset_length > length:
                break
            else:
                index_list.append(index)
                subset_length += 1
        
        self.balanced = balanced
        self.dataset = dataset
        self.classes = classes
        self.indices = functools.reduce(operator.add, classes.values(), [])
        # print(self.classes)
        # print({c: len(i) for c, i in self.classes.items()})

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def train_validate_full(
    *,
    model: torch.nn.Module,
    seed: int = SEED,
    dataset_class: type(torch.utils.data.Dataset) = CIFAR10_DATASET_CLASS,
    train_dataset: typing.Optional[torch.utils.data.Dataset] = None,
    test_dataset: typing.Optional[torch.utils.data.Dataset] = None,
    transform: torchvision.transforms.Compose = CIFAR10_TRANSFORM,
    dataset_path: os.PathLike = CIFAR10_PATH,
    batch_size: int = CIFAR10_BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    model_checkpoint_path: os.PathLike = MODEL_CHECKPOINT_PATH,
    model_checkpoint_basename: str = MODEL_CHECKPOINT_BASENAME,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = LOSS_FN,
    optimizer_class: typing.Callable[[torch.Tensor, float], torch.optim.Optimizer] = OPTIMIZER_CLASS,
    device: torch.device = DEVICE,
    training_log_file: os.PathLike = TRAINING_LOG_FILE,
    validation_log_file: os.PathLike = VALIDATION_LOG_FILE,
    transform_target_based_on_input: typing.Optional[typing.Callable] = None,
    scheduler_class=None,
    **kwargs,
) -> typing.Dict[str, float]:
    dataset_path = pathlib.Path(dataset_path)
    training_log_file = pathlib.Path(training_log_file)
    validation_log_File = pathlib.Path(validation_log_file)

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dataset = dataset_class(str(dataset_path), train=True, transform=transform) if train_dataset is None else train_dataset 
    test_dataset = dataset_class(str(dataset_path), train=False, transform=transform) if test_dataset is None else test_dataset

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    scheduler = scheduler_class(optimizer) if scheduler_class is not None else None
    optimizer.zero_grad()

    model = model.to(device)

    model.train()
    training_accuracy = 0
    validation_accuracy = 0
    training_log_file.parent.mkdir(parents=True, exist_ok=True)
    training_log = training_log_file.open("w")
    validation_log_file.parent.mkdir(parents=True, exist_ok=True)
    validation_log = validation_log_file.open("w")
    for e in range(1, epochs + 1):
        training_log.write(f"Epoch {e}\n")

        if scheduler is not None:
            scheduler.step()
            training_log.write(f"Current Learning Rate: {scheduler.get_lr()}\n")

        epoch_training_accuracy = train_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch_training_log=training_log,
            transform_target_based_on_input=transform_target_based_on_input,
        )

        training_accuracy = (training_accuracy * (e - 1) + epoch_training_accuracy) / e

        model_checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(model_checkpoint_path / model_checkpoint_basename) + '.' + str(e))

        validation_log.write(f"Epoch {e}\n")

        epoch_validation_accuracy = validate_epoch(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            epoch_validation_log=validation_log,
        )

        validation_accuracy = (validation_accuracy * (e - 1) + epoch_validation_accuracy) / e

    return {"training_accuracy": training_accuracy, "validation_accuracy": validation_accuracy}


def train_epoch(
    *,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_training_log: typing.TextIO,
    transform_target_based_on_input: typing.Optional[typing.Callable] = None,
) -> float:
    optimizer.zero_grad()

    model = model.to(device)
        
    model.train()
    total = 0
    correct = 0
    for train_batch in iter(dataloader):
        x, y_target = train_batch
        x = x.to(device)
        y_target = y_target.to(device)
        x, y_target = convert_batch(x, y_target, transform_target_based_on_input=transform_target_based_on_input)
        y = model(x)
        y_indices = torch.argmax(y, dim=1)
        loss = loss_fn(y, y_target)
        total += y_target.size(0)
        correct += (y_indices == y_target).sum().item()
        epoch_training_log.write(f"train loss: {loss.item()}\n")
        epoch_training_log.write(f"train accuracy: {correct / total}\n")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct / total


def validate_epoch(
    *,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    epoch_validation_log: typing.TextIO,
) -> float:
    model = model.to(device)
        
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for val_batch in iter(dataloader):
            x, y_target = val_batch
            x = x.to(device)
            y_target = y_target.to(device)
            y = model(x)
            y_indices = torch.argmax(y, dim=1)
            loss = loss_fn(y, y_target)
            total += y_target.size(0)
            correct += (y_indices == y_target).sum().item()
            epoch_validation_log.write(f"validation loss: {loss.item()}\n")
            epoch_validation_log.write(f"validation accuracy: {correct / total}\n")

    return correct / total


def convert_batch(
    input_: torch.Tensor,
    target: torch.Tensor,
    transform_target_based_on_input: typing.Optional[typing.Callable] = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    if transform_target_based_on_input is None:
        return input_, target
    return input_, transform_target_based_on_input(input_)

