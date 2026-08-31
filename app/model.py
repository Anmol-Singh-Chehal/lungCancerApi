import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# LungCancerNet
# ResNet152 + AlexNet
# ============================================================

class LungCancerNet(nn.Module):

    def __init__(self, num_classes, freeze_backbones=False):
        super().__init__()

        # ----------------------------------------------------
        # ResNet152
        # ----------------------------------------------------

        resnet = models.resnet152(
            weights=None
        )

        self.resnet = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.resnet_dim = 2048

        # ----------------------------------------------------
        # AlexNet
        # ----------------------------------------------------

        alexnet = models.alexnet(
            weights=None
        )

        self.alexnet = alexnet.features

        self.alex_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.alex_dim = 256

        # ----------------------------------------------------
        # Freeze Backbones
        # ----------------------------------------------------

        if freeze_backbones:

            for p in self.resnet.parameters():
                p.requires_grad = False

            for p in self.resnet[-1].parameters():
                p.requires_grad = True

            for p in self.alexnet.parameters():
                p.requires_grad = False

            for p in self.alexnet[-1].parameters():
                p.requires_grad = True

        # ----------------------------------------------------
        # ResNet Head
        # ----------------------------------------------------

        self.resnet_head = nn.Sequential(

            nn.Linear(2048, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # AlexNet Head
        # ----------------------------------------------------

        self.alex_head = nn.Sequential(

            nn.Linear(256, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        fused_dim = 512

        self.classifier = nn.Sequential(

            nn.Linear(fused_dim, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(256, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward_features(self, x):

        resnet_feat = self.resnet(x)

        resnet_feat = self.resnet_pool(
            resnet_feat
        )

        resnet_feat = torch.flatten(
            resnet_feat,
            1
        )

        resnet_feat = self.resnet_head(
            resnet_feat
        )

        alex_feat = self.alexnet(x)

        alex_feat = self.alex_pool(
            alex_feat
        )

        alex_feat = torch.flatten(
            alex_feat,
            1
        )

        alex_feat = self.alex_head(
            alex_feat
        )

        fused = torch.cat(
            (
                resnet_feat,
                alex_feat
            ),
            dim=1
        )

        return fused

    def forward(self, x):

        features = self.forward_features(x)

        return self.classifier(features)