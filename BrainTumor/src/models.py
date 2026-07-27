import torchvision.models as models
import torch.nn as nn

from config import NUM_CLASSES
from utils import count_parameters, count_trainable_parameters


def get_model(name, pretrained=True):

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )

    elif name == "vit":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, NUM_CLASSES
        )

    elif name == "swin":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    elif name == "coatnet":
        import timm
        try:
            model = timm.create_model(
                "coatnet_0_rw_224", pretrained=pretrained,
                num_classes=NUM_CLASSES
            )
        except RuntimeError:
            coat_models = [m for m in timm.list_models() if "coatnet" in m]
            if not coat_models:
                raise RuntimeError(
                    "No se encontro ningun modelo CoAtNet en timm. "
                    "Actualiza timm: pip install -U timm>=0.9.0"
                )
            model = timm.create_model(
                coat_models[0], pretrained=pretrained,
                num_classes=NUM_CLASSES
            )
            print(f"Usando {coat_models[0]} como alternativa a coatnet_0")

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choices: resnet50, efficientnet_b0, vit, swin, coatnet"
        )

    return model


def print_model_info(model, model_name):
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    print(f"\nModelo: {model_name}")
    print(f"Parametros totales:     {total:,}")
    print(f"Parametros entrenables: {trainable:,}\n")
    return {"model": model_name, "total_params": total, "trainable_params": trainable}
