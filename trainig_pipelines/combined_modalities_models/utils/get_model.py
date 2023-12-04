import os
import sys

import torch
from torch import nn
from torch.nn import functional as F

# Add the trainig_pipelines/image_models/models directory to the system path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    ),
)
from multimodal_model import MultiModalFastAi


def get_model(
        model_type: str,
        num_classes: int = 8
):
    """
    Get model from str_model.

    Parameters:
    - model_type can be:    MultiModalFastAi

    - num_classes:          amount of classes in target.
    """

    # ANSI escape code for bold and green text
    bold_green_code = "\033[1;92m"

    # Reset ANSI escape code
    reset_code = "\033[0m"

    print(
        "Model type: "
        + bold_green_code
        + model_type
        + reset_code
    )

    if model_type == "MultiModalFastAi":
        model = MultiModalFastAi(
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Inputted model: {model_type} not implemented.")

    total_params = sum(p.numel() for p in model.parameters())
    print('     Number of parameters:', total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('     Number of trainable parameters:', trainable_params, '\n')

    return model
