from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms


def get_data_loader(name, batch_size, download=True):
    dataset = get_dataset(dataset=name, download=download)

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )

    val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )

    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )

    # Prepare the standard data loader
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size)
    val_loader = get_eval_loader("standard", val_data, batch_size=1)
    test_loader = get_eval_loader("standard", test_data, batch_size=1)

    return dataset, train_loader, val_loader, test_loader
