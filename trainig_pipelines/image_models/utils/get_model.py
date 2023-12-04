import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# Add the trainig_pipelines/image_models/models directory to the system path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    ),
)
from inception_resnet import InceptionResnetV1, InceptionResnetV1FastAI, InceptionResnetV1Encoder


def get_model(
        model_type: str,
        num_classes: int = 8,
        is_multilabel: bool = False,
        is_fastai_head: bool = False,
        pretrained: str = 'vggface2',
        freeze_all_except_last: bool = False,
        unfreeze_first: bool = False
):
    """
    Get model from str_model.

    Parameters:
    - model_type can be:    InceptionResnetV1, InceptionResnetV1FastAI, densenet121

    - num_classes:          amount of classes in target.
    - image_size:           size of the input image data.
    - input_data_channels:  amount of channels in the input data.
    """

    # ANSI escape code for bold and green text
    bold_green_code = "\033[1;92m"

    # Reset ANSI escape code
    reset_code = "\033[0m"

    if is_fastai_head is True and model_type == "InceptionResnetV1":
        model_type = "InceptionResnetV1FastAI"

    print(
        "Model type: "
        + bold_green_code
        + model_type
        + reset_code
        + " is_multilabel: "
        + bold_green_code
        + str(is_multilabel)
        + reset_code
    )

    if model_type == "InceptionResnetV1":
        model = InceptionResnetV1(
            pretrained=pretrained,
            classify=True,
            num_classes=num_classes,
            is_multilabel=is_multilabel)

    elif model_type == "InceptionResnetV1Encoder":
        model = InceptionResnetV1Encoder(
            pretrained=pretrained,
            classify=True,
            num_classes=num_classes,
            is_multilabel=is_multilabel)

    elif model_type == "InceptionResnetV1FastAI":
        model = InceptionResnetV1FastAI(
            pretrained=pretrained,
            classify=True,
            num_classes=num_classes,
            is_multilabel=is_multilabel)

    elif model_type == "densenet121":
        model = models.densenet121(weights='IMAGENET1K_V1')

        # for param in model.parameters():
        #     param.requires_grad = False

        if is_fastai_head:
            model.classifier = nn.Sequential(
                nn.Linear(in_features=1024, out_features=512, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Dropout(p=0.5, inplace=False),

                nn.Linear(in_features=512, out_features=num_classes, bias=False)
            )
        else:
            model.classifier = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(), nn.Linear(512, num_classes))

    else:
        raise ValueError(f"Inputted model: {model_type} not implemented.")

    if freeze_all_except_last:
        # Freeze all layers except the linear one
        model.freeze_layers(
            freeze_all_except_last=True,
            unfreeze_first=False
        )

    if unfreeze_first:
        # Freeze all layers except the linear one
        model.freeze_layers(
            freeze_all_except_last=False,
            unfreeze_first=True
        )

    total_params = sum(p.numel() for p in model.parameters())
    print('     Number of parameters:', total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('     Number of trainable parameters:', trainable_params, '\n')

    return model

# net = get_model(
#     model_type = 'InceptionResnetV1',
#     num_classes = 8,
#     is_multilabel = True,
#     freeze_all_except_last = True,
#     unfreeze_first = False
# )
