import torch.nn as nn
import torch
import torchvision.models as models


class CEVTModel(nn.Module):
    def __init__(self, dataset, feature_extractor='resnet18', output_layer=102):
        super(CEVTModel, self).__init__()

        source_index, source_data, source_label, target_index, target_data, target_label = dataset[0]
        # Load the desired feature extractor
        if feature_extractor == 'resnet18':
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.classifier = nn.Linear(512, output_layer)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

    def forward(self, X):
        b, frames, c, h, w = X.size()
        X = X.view(b * frames, c, h, w)
        features = self.feature_extractor(X)  # features size b*frames, features
        features = features.view(b, frames, -1)
        features = torch.mean(features, dim=1)  # average pooling along temporal dimension
        features = self.classifier(features)
        return features