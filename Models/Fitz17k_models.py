from torchvision import models
from torch import nn


class Fitz17kResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes)
        # self.feature_extractor = nn.Sequential(
        #     *list(self.feature_extractor.children())[:-1]
        # )
        # self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        output = self.feature_extractor(x)
        # output = output.view(x.size(0), -1)
        # output = self.classifier(output)
        return output
