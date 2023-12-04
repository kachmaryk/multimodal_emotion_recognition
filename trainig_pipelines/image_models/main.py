import os
import sys
import json
import wandb
import random
import argparse
import numpy as np

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

# Add the ml_part/models_tryouts/second_model directory to the system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from training import Trainer

# Add the ml_part/models_tryouts/second_model/utils directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))
from dataloaders import MyDataLoader
from get_model import get_model
from readFile import readFile

from torchviz import make_dot
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


def set_seed(seed_value: int) -> None:
    """
    Set seed for reproducibility.
    """

    # Python built-in RNG
    random.seed(seed_value)

    # Numpy RNG
    np.random.seed(seed_value)

    # PyTorch RNGs
    torch.manual_seed(seed_value)

    # If you're using CUDA, and want to ensure reproducibility on GPU as well
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed_value)


def main(exp_config) -> None:
    # Set seed if needed
    if exp_config.is_set_seed:
        set_seed(exp_config.seed_value)

    if exp_config.data_path is None:
        raise ValueError(f"data_path should be specified, got data_path = {exp_config.data_path}")

    checkpoint_folder = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    data_path = exp_config.data_path
    num_classes = exp_config.num_classes
    is_multilabel = exp_config.is_multilabel
    is_fastai_head = exp_config.is_fastai_head

    model_type = exp_config.model_type
    pretrained = exp_config.pretrained
    freeze_all_except_last = exp_config.freeze_all_except_last
    unfreeze_first = exp_config.unfreeze_first

    batch_size = exp_config.batch_size
    epochs = exp_config.epochs
    lr = exp_config.lr

    train_loader, validation_loader, test_loader = MyDataLoader(
        data_path=data_path,
        batch_size=batch_size,
        num_classes=num_classes,
        is_multilabel=is_multilabel
    )

    net = get_model(
        model_type=model_type,
        num_classes=num_classes,
        is_multilabel=is_multilabel,
        is_fastai_head=is_fastai_head,
        pretrained=pretrained,
        freeze_all_except_last=freeze_all_except_last,
        unfreeze_first=unfreeze_first
    )
    print(net)

    # Visualization of the net
    batch = next(iter(train_loader))
    inputs, labels, *metadata = batch
    yhat = net(inputs)

    make_dot(yhat, params=dict(list(net.named_parameters()))).render("inception_resnetv1_fastai_head", format="png")

    input_names = ['Input image']
    output_names = ['Predicted class']
    torch.onnx.export(net, inputs, 'inception_resnetv1_fastai_head.onnx', input_names=input_names,
                      output_names=output_names)

    # all_labels = []
    # for val_iter, data in enumerate(train_loader, 0):
    #     inputs, labels, *metadata = data
    #     all_labels.extend([lab.item() for lab in labels])
    #
    #     # for img in inputs:
    #     #     print(img.shape)
    #     #     plt.imshow(img.numpy())
    #     #     plt.show()
    #
    # print(np.unique(all_labels, return_counts=True))

    wandb.init(
        # Set the project where this run will be logged
        project="diploma-images-data",

        # Track hyperparameters and run metadata
        config={
            "data_path": os.path.basename(data_path),
            "num_classes": num_classes,
            "is_multilabel": is_multilabel,
            "is_fastai_head": is_fastai_head,

            "model_type": model_type,
            "pretrained": pretrained,
            "freeze_all_except_last": freeze_all_except_last,
            "unfreeze_first": unfreeze_first,

            "optimizer_type": exp_config.optimizer_type,
            "scheduler_type": exp_config.scheduler_type,

            "batch_size": batch_size,
            "lr": lr,
            "patience": exp_config.patience
        }
    )
    # wandb.config.update(exp_config, allow_val_change=True)
    wandb.watch(net)

    # Initialize optimizers
    if exp_config.optimizer_type == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=exp_config.momentum,
                              weight_decay=exp_config.weight_decay)
        cycle_momentum = True
    elif exp_config.optimizer_type == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr,
                               weight_decay=exp_config.weight_decay,
                               betas=(0.5, 0.999))
        cycle_momentum = False
    else:
        raise NotImplementedError(f"There is no such option for optimizer such as: {exp_config.optim}")

    scheduler_step_size = int(np.floor(len(train_loader) / 2))

    """ Train model """
    trainer = Trainer(net, optimizer, epochs=epochs,
                      scheduler_type=exp_config.scheduler_type,
                      scheduler_step_size=scheduler_step_size,
                      cycle_momentum=cycle_momentum,
                      patience=exp_config.patience,
                      use_cuda=exp_config.use_cuda,
                      gpu_num=exp_config.gpu_num,
                      checkpoint_folder=checkpoint_folder,
                      num_classes=num_classes)

    trainer.train_model(train_loader, validation_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--is_set_seed', type=bool, default=True)
    parser.add_argument('--seed_value', type=int, default=1656079)

    parser.add_argument('--data_path', type=str, default=None,
                        help="Folder which contains training data")
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Based on amount of classes data was distributed')
    parser.add_argument('--is_multilabel', type=bool, default=False,
                        help='If model prediction & ground truth is multi-label data')
    parser.add_argument('--is_fastai_head', type=bool, default=False,
                        help='If model prediction layer is fastai_head')

    parser.add_argument('--model_type', type=str, default='densenet121',
                        help='InceptionResnetV1 | densenet121')
    parser.add_argument('--pretrained', type=str, default='vggface2',
                        help='Type of pretrained model')
    parser.add_argument('--freeze_all_except_last', type=bool, default=True,
                        help='Freeze all layers except linear in model')
    parser.add_argument('--unfreeze_first', type=bool, default=True,
                        help='Unfreeze initial Conv layers in the model')

    parser.add_argument('--optimizer_type', type=str, default="Adam")
    parser.add_argument('--scheduler_type', type=str, default="CyclicLR",
                        help="possible values: CyclicLR | ReduceLROnPlateau")

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=60)

    parser.add_argument('--TextArgs', type=str,
                        default='configs/major_emotion_cropped_face_single_test_unfreeze_all_cyclic_lr.txt',
                        help='Path to text with training settings')

    parse_list = readFile(parser.parse_args().TextArgs)
    _config = parser.parse_args(parse_list)

    main(exp_config=_config)
