import torch
import torch.nn as nn
import torchvision.models as models


class VggSimCLR(nn.Module):
    def __init__(self, base_model, pretrained, num_classes, out_dim=128):
        super().__init__()
        if pretrained:
            self.vgg_dict = {"vgg11": models.vgg11(pretrained=True, num_classes=1000),
                             "vgg19": models.vgg19(pretrained=True, num_classes=1000)}
        else:
            self.vgg_dict = {"vgg11": models.vgg11(pretrained=False, num_classes=num_classes),
                             "vgg19": models.vgg19(pretrained=False, num_classes=num_classes)}

        vgg = self._get_basemodel(base_model)
        self.in_dim_mlp = vgg.classifier[0].in_features  # 25088
        self.out_dim_mlp = vgg.classifier[-1].in_features  # 4096

        # need to separate VGG's backbone and fc
        self.backbone = nn.Sequential(*(list(vgg.children())[:-1]))  # backbone
        self.classify_head = nn.Sequential(
            nn.Linear(self.in_dim_mlp, self.out_dim_mlp),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.out_dim_mlp, self.out_dim_mlp),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.out_dim_mlp, num_classes),
        )

        # add mlp projection head
        self.contrastive_head = nn.Sequential(nn.Linear(self.in_dim_mlp, self.out_dim_mlp), nn.ReLU(),
                                              nn.Linear(self.out_dim_mlp, out_dim))

    def _get_basemodel(self, model_name):
        try:
            model = self.vgg_dict[model_name]
        except KeyError:
            raise Exception(
                "Invalid backbone architecture. Check the config file and pass one of: vgg11 or vgg19")
        else:
            return model

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        cla_out = self.classify_head(x)
        con_out = self.contrastive_head(x)

        return cla_out, con_out