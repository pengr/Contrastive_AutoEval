import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, pretrained, num_classes, out_dim=128):
        super(ResNetSimCLR, self).__init__()
        if pretrained:  # num_classes=out_dim just use for contrastive head
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=True, num_classes=1000),
                                "resnet34": models.resnet34(pretrained=True, num_classes=1000),
                                "resnet50": models.resnet50(pretrained=True, num_classes=1000)}
        else:
            self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=num_classes),
                                "resnet34": models.resnet34(pretrained=False, num_classes=num_classes),
                                "resnet50": models.resnet50(pretrained=False, num_classes=num_classes)}

        resnet = self._get_basemodel(base_model)
        self.dim_mlp = resnet.fc.in_features

        # need to separate ResNet's backbone and fc
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))  # backbone
        self.classify_head = nn.Linear(self.dim_mlp, num_classes)      # replace the original fc of ResNet

        # add mlp projection head
        self.contrastive_head = nn.Sequential(nn.Linear(self.dim_mlp, self.dim_mlp), nn.ReLU(), nn.Linear(self.dim_mlp, out_dim))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception("Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        cla_out = self.classify_head(x)
        con_out = self.contrastive_head(x)
        return cla_out, con_out