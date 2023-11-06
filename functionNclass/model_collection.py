import torch.nn as nn
import torch
import torchvision.models as models

class CEVTModel(nn.Module):
    def __init__(self, dataset, feature_extractor='resnet18', output_layer=102):
        super(CEVTModel, self).__init__()

        #source_index, source_data, source_label = dataset[0]
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

class CEVTModel_old(nn.Module):
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


# -> feed into known/ unknown classifier -> the known classifier we pass the known classes to

class Adapter(nn.Module):
    def __init__(self, config, input_dim=2034, output_dim=512):
        super(Adapter, self).__init__()
        hidden_dim = input_dim // config['reduction']
        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config['dropout']),
        ]

        for _ in range(config['n_layers'] - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=config['dropout']))

        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.fc = nn.Sequential(*layers)


    def forward(self, x):
        x = self.fc(x)
        return x


class DualClassifier_sharedLayer(nn.Module):
    def __init__(self, dataset, feature_extractor='resnet18', num_classes=102, config=None):
        super(DualClassifier_sharedLayer, self).__init__()

        # Extract a sample from the dataset to understand its structure
        source_index, source_data, source_label, target_index, target_data, target_label = dataset[0]

        # Define the feature extractor (e.g., ResNet18) -> we need to freeze
        if feature_extractor == 'resnet18':
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            for param in self.feature_extractor.parameters():  # Freezing the backbone
                param.requires_grad = False
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

        self.adapter = Adapter(config, input_dim=512, output_dim=256)
        self.label_classifier = nn.Linear(256, num_classes)
        self.known_unknown_classifier = nn.Linear(256, 1)

        # Freeze feature_extractor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        # # Make sure Adapter has requires_grad=True
        # for param in self.adapter.parameters():
        #     param.requires_grad = True

        # # Make sure label_classifier has requires_grad=True
        # for param in self.label_classifier.parameters():
        #     param.requires_grad = True

        # # Make sure known_unknown_classifier has requires_grad=True
        # for param in self.known_unknown_classifier.parameters():
        #     param.requires_grad = True



    def get_features(self, X):
        # Reshape the input to match the expected shape of the feature extractor
        b, frames, c, h, w = X.size()
        X_obs = X.view(b * frames, c, h, w)

        # Extract features using the selected feature extractor (e.g., ResNet)
        features = self.feature_extractor(X_obs)
        features = features.view(b, frames, -1)
        features = torch.mean(features, dim=1)  # Average pooling along the temporal dimension
        adapted_features = self.adapter(features)  # Pass features through Adapter
        #print("Shape of adapted_features:", adapted_features.shape)

        return adapted_features


    def forward(self, X, known_unknown_binary_mask = [], mode='Train'):
        #print('########################## INSIDE THE MODEL ################################### DOWN')
        # print("Shape of X:", X.shape)

        adapted_features = self.get_features(X)
        known_unknown_score = self.known_unknown_classifier(adapted_features)

        # print("Data type of known_unknown_score:", known_unknown_score.dtype)
        # print("Shape of known_unknown_score:", known_unknown_score.shape)

        # Apply the sigmoid activation function to get the binary vector
        probabilities = torch.sigmoid(known_unknown_score)

        # Apply a threshold (e.g., 0.5) to classify into True (KNOWN) or 0 (UNKNOWN)
        known_unknown_labels = (probabilities >= 0.5) # [8,1]

        if mode =='Train':
          adapted_features_known = adapted_features[known_unknown_binary_mask]

        elif mode == 'Test':
          # Check if all values in known_unknown_labels are False
          if torch.sum(known_unknown_labels.float()) == 0:
              #print('known_unknown_labels.requires_grad: ',known_unknown_labels.requires_grad)
              return None, known_unknown_labels.float()  # Return None for known_labels

          #print("Shape of known_unknown_labels:", known_unknown_labels.shape)
          adapted_features_known = adapted_features[known_unknown_labels.squeeze()]  #adapted_features = batch * 256

        known_labels = self.label_classifier(adapted_features_known)

        # #known_unknown_labels.requires_grad_()
        # print("Shape of adapted_features:", adapted_features_known.shape)
        # print("Shape of known_labels:", known_labels.shape)
        # print(known_unknown_binary_mask.device)
        # print(adapted_features.device)
        # print('known_labels.requires_grad: ',known_labels.requires_grad)
        # print('known_unknown_labels.requires_grad: ',known_unknown_labels.requires_grad)
        #print('########################## INSIDE THE MODEL ################################### UP')


        return known_labels, known_unknown_labels.float()  # binary cross entropy does not accept Bolean vectors.        