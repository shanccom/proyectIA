import torchvision.models as models
import torch.nn as nn

from config import NUM_CLASSES
from utils import count_parameters, count_trainable_parameters


def get_model(name):

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )

    elif name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, NUM_CLASSES
        )

    elif name == "swin":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    elif name == "coatnet":
        import timm
        model = timm.create_model(
            "coatnet_0", pretrained=True, num_classes=NUM_CLASSES
        )

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choices: resnet50, efficientnet_b0, vit, swin, coatnet"
        )

    return model


def print_model_info(model, model_name):
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    print(f"\n{'='*50}")
    print(f"Modelo: {model_name}")
    print(f"Parámetros totales:   {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")
    print(f"{'='*50}\n")
    return {"model": model_name, "total_params": total, "trainable_params": trainable}
