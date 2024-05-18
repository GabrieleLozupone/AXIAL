# --------------------------------------------------------------------------------
# Copyright (c) 2024 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
# 
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
# 
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------
import torch
import torchvision
from torch import nn


def get_backbone(model_name: str,
                 device: torch.device,
                 radimgnet_classes: int = 116,
                 pretrained_on: str = "ImageNet"):
    """
    Load a pretrained backbone on RadImageNet from path.
    The .pth file must be in the models/backbone/ directory and the name must be model_name.pth.
    Args:
        - model_name (str): name of the model to load.
            Expected values: "EfficientNetV2S", "EfficientNetV2M", "EfficientNetB1", "EfficientNetB4", "EfficientNetB6",
             "ResNet50", "ResNet34", "ResNet101", "SwinV2T", "DenseNet121", "DenseNet161", "VGG16", "VGG19".
        - radimgnet_classes (int): number of classes of RadImageNet that the pretrained model was trained on.
        - device (torch.device): device to load the model on.
        - pretrained_on (str): name of the dataset to load. Expected values: "RadImageNet", "ImageNet". If different
          values are passed, a not pretrained backbone will be returned.
    Returns:
        - backbone (nn.Module): the backbone.
        - embedding_dim (int): the embedding dimension of the backbone.
    """
    # Initialize variables
    backbone = None
    embedding_dim = None
    # Check if a valid model name was passed
    if model_name not in ["EfficientNetV2S", "EfficientNetV2M", "EfficientNetB1", "EfficientNetB4", "EfficientNetB6",
                          "ResNet50", "ResNet34", "ResNet101", "SwinV2T", "DenseNet121", "DenseNet161",
                          "VGG11", "VGG13", "VGG16", "VGG19", "ConvNextBase", "ConvNextSmall", "ConvNextTiny"]:
        raise ValueError(f"Invalid model name: {model_name}")
    if model_name == "ConvNextBase":
        # Set the embedding dimension according to the model
        embedding_dim = 1024
        # Load model structure from torchvision
        model = torchvision.models.convnext_base()
        if pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.convnext_base(weights=weights)
        backbone = model.features
    if model_name == "ConvNextSmall":
        # Set the embedding dimension according to the model
        embedding_dim = 768
        # Load model structure from torchvision
        model = torchvision.models.convnext_small()
        if pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.convnext_small(weights=weights)
        backbone = model.features
    if model_name == "ConvNextTiny":
        # Set the embedding dimension according to the model
        embedding_dim = 768
        # Load model structure from torchvision
        model = torchvision.models.convnext_tiny()
        if pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.convnext_tiny(weights=weights)
        backbone = model.features
    if model_name == "SwinV2T":
        # Set the embedding dimension according to the model
        embedding_dim = 768
        # Load model structure from torchvision
        model = torchvision.models.swin_v2_t()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("SwinV2T pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.Swin_V2_T_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.swin_v2_t(weights=weights)
        backbone = nn.Sequential(
            model.features,
            model.permute
        )
    elif model_name == "VGG11":
        # Set the embedding dimension according to the model
        embedding_dim = 512
        # Load model structure from torchvision
        model = torchvision.models.vgg11_bn()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("VGG11 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.vgg11_bn(weights=weights)
        backbone = model.features
    elif model_name == "VGG13":
        # Set the embedding dimension according to the model
        embedding_dim = 512
        # Load model structure from torchvision
        model = torchvision.models.vgg13_bn()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("VGG13 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.VGG13_BN_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.vgg13_bn(weights=weights)
        backbone = model.features
    elif model_name == "VGG16":
        # Set the embedding dimension according to the model
        embedding_dim = 512
        # Load model structure from torchvision
        model = torchvision.models.vgg16_bn()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("VGG16 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.vgg16_bn(weights=weights)
        backbone = model.features
    elif model_name == "VGG19":
        # Set the embedding dimension according to the model
        embedding_dim = 512
        # Load model structure from torchvision
        model = torchvision.models.vgg19_bn()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("VGG19 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.vgg19_bn(weights=weights)
        backbone = model.features
    elif model_name == "DenseNet121":
        # Set the embedding dimension according to the model
        embedding_dim = 1024
        # Load model structure from torchvision
        model = torchvision.models.densenet121()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("DenseNet121 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.densenet121(weights=weights)
        backbone = model.features
    elif model_name == "DenseNet161":
        # Set the embedding dimension according to the model
        embedding_dim = 2208
        # Load model structure from torchvision
        model = torchvision.models.densenet161()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("DenseNet161 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.densenet161(weights=weights)
        backbone = model.features
    elif model_name == "EfficientNetB1":
        # Set the embedding dimension according to the model
        embedding_dim = 1280
        # Load model structure from torchvision
        model = torchvision.models.efficientnet_b1()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("EfficientNetB1 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
            # Pass the weights to the model
            model = torchvision.models.efficientnet_b1(weights=weights)
        backbone = model.features
    elif model_name == "EfficientNetB4":
        # Set the embedding dimension according to the model
        embedding_dim = 1792
        # Load model structure from torchvision
        model = torchvision.models.efficientnet_b4()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("EfficientNetB4 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.efficientnet_b4(weights=weights)
        backbone = model.features
    elif model_name == "EfficientNetB6":
        # Set the embedding dimension according to the model
        embedding_dim = 2304
        # Load model structure from torchvision
        model = torchvision.models.efficientnet_b6()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("EfficientNetB6 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.efficientnet_b6(weights=weights)
        backbone = model.features
    elif model_name == "EfficientNetV2S":
        # Set the embedding dimension according to the model
        embedding_dim = 1280
        # Load model structure from torchvision
        model = torchvision.models.efficientnet_v2_s()
        if pretrained_on == "RadImageNet":
            # Change the last layer to match the number of classes in the saved model
            model.classifier = nn.Sequential(
                nn.Dropout(0.4, inplace=True),
                nn.Linear(embedding_dim, radimgnet_classes),
            )
            # Load the pretrained weights on RadImageNet
            model.load_state_dict(torch.load('models/backbone/EfficientNetV2S.pth', map_location=device))
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.efficientnet_v2_s(weights=weights)
        backbone = model.features
    elif model_name == "EfficientNetV2M":
        # Set the embedding dimension according to the model
        embedding_dim = 1280
        # Load model structure from torchvision
        model = torchvision.models.efficientnet_v2_m()
        if pretrained_on == "RadImageNet":
            # Change the last layer to match the number of classes in the saved model
            model.classifier = nn.Sequential(
                nn.Dropout(0.4, inplace=True),
                nn.Linear(embedding_dim, radimgnet_classes),
            )
            # Load the pretrained weights on RadImageNet
            model.load_state_dict(torch.load('models/backbone/EfficientNetV2M.pth', map_location=device))
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.efficientnet_v2_m(weights=weights)
        backbone = model.features
    elif model_name == "ResNet34":
        # Set the embedding dimension according to the model
        embedding_dim = 512
        # Load model structure from torchvision
        model = torchvision.models.resnet34()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("ResNet34 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            # Pass the weights to the model
            model = torchvision.models.resnet34(weights=weights)
        backbone = nn.Sequential(*list(model.children())[:-2])
    elif model_name == "ResNet50":
        # Set the embedding dimension according to the model
        embedding_dim = 2048
        # Load model structure from torchvision
        model = torchvision.models.resnet50()
        if pretrained_on == "RadImageNet":
            # Change the last layer to match the number of classes in the saved model
            model.fc = nn.Linear(embedding_dim, 1)
            # Load the pretrained weights on RadImageNet
            model.load_state_dict(torch.load('models/backbone/RadImageNet-ResNet50.pth', map_location=device))
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            # Pass the weights to the model
            model = torchvision.models.resnet50(weights=weights)
        backbone = nn.Sequential(*list(model.children())[:-2])
    elif model_name == "ResNet101":
        # Set the embedding dimension according to the model
        embedding_dim = 2048
        # Load model structure from torchvision
        model = torchvision.models.resnet101()
        if pretrained_on == "RadImageNet":
            raise NotImplementedError("ResNet101 pretrained on RadImageNet is not available yet")
        elif pretrained_on == "ImageNet":
            # Load the pretrained weights from torchvision
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
            # Pass the weights to the model
            model = torchvision.models.resnet101(weights=weights)
        backbone = nn.Sequential(*list(model.children())[:-2])
    if backbone is None:
        raise ValueError(f"Invalid model name: {model_name}")
    return backbone, embedding_dim
