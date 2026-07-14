import torchvision.models as models
import torch.nn as nn

from config import NUM_CLASSES


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

    elif name == "cvt":
        # Requires: pip install cvt-pytorch
        from cvt import CVT
        model = CVT(num_classes=NUM_CLASSES)

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choices: resnet50, efficientnet_b0, vit, swin, cvt"
        )

    return model
