import os
import random

import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler


class MyDataset:
    """
    Class to store a given dataset.

    Parameters:
    - samples: list of tensor images
    - transform: data transforms for augmentation
    """

    def __init__(self, samples, num_classes, is_multilabel, transform=None):
        self.num_samples = len(samples)
        self.data = samples
        self.transform = transform
        self.num_classes = num_classes
        self.is_multilabel = is_multilabel

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label, *metadata = self.data[idx]

        if self.is_multilabel:
            label = torch.tensor(label)
        else:
            label = torch.argmax(torch.tensor(label))

        if self.transform:
            img = self.transform(img)

        return img.float(), label, metadata


def MyDataLoader(data_path, batch_size, num_classes,
                 is_multilabel, num_workers=1):
    print("\n----Loading dataset----")
    TRAIN_TRANSFORM_IMG = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0), antialias=True),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])
    VAL_TRANSFORM_IMG = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

    training = torch.load(os.path.join(data_path, "train_dataset.pt"))
    validation = torch.load(os.path.join(data_path, "val_dataset.pt"))
    test = torch.load(os.path.join(data_path, "test_dataset.pt"))

    train_dataset = MyDataset(
        training, num_classes=num_classes,
        is_multilabel=is_multilabel,
        transform=TRAIN_TRANSFORM_IMG
    )
    validation_dataset = MyDataset(
        validation, num_classes=num_classes,
        is_multilabel=is_multilabel,
        transform=VAL_TRANSFORM_IMG
    )
    test_dataset = MyDataset(
        test, num_classes=num_classes,
        is_multilabel=is_multilabel,
        transform=VAL_TRANSFORM_IMG
    )

    weights = compute_sample_weights(dataset=training, num_classes=num_classes)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    print(" Training images: ", len(train_dataset))
    print(" Validation images: ", len(validation_dataset))
    print(" Test images: ", len(test_dataset))
    print("-------------------------\n")

    return train_loader, validation_loader, test_loader


def compute_sample_weights(dataset, num_classes: int):
    label_counts = {i: 0 for i in range(max(2, num_classes))}

    # Calculate label frequencies
    for _, label, *_ in dataset:
        class_idx = torch.argmax(torch.tensor(label)).item()
        label_counts[class_idx] += 1

    adjustment_factors = {0: 1.05,
                          1: 1.05,
                          2: 1.05,
                          3: 1.0,
                          4: 1.05,
                          5: 1.05,
                          6: 1.05,
                          7: 1.05,
                          8: 1.05}

    # Calculate weights for each class with adjustments
    class_weights = {k: (1 / v) * adjustment_factors[k] for k, v in label_counts.items()}


    # Compute sample weights
    sample_weights = []
    for _, label, *_ in dataset:
        class_idx = torch.argmax(torch.tensor(label)).item()

        sample_weight = class_weights[class_idx]
        sample_weights.append(sample_weight)

    return sample_weights
