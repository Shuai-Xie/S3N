from torch import nn
from torchvision import models
from nest import register


class FT_Resnet(nn.Module):
    def __init__(self, mode='resnet50', fc_or_fcn='fc', num_classes=10, pretrained=True):
        super(FT_Resnet, self).__init__()

        if mode == 'resnet50':
            model = models.resnet50(pretrained=pretrained)  # pretrain on ImageNet
        elif mode == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif mode == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            model = models.resnet18(pretrained=pretrained)

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4  # conv5_x
        )
        self.num_classes = num_classes
        self.num_features = model.layer4[1].conv1.in_channels
        self.fc_or_fcn = fc_or_fcn
        if self.fc_or_fcn == 'fc':
            self.classifier = nn.Linear(self.num_features, self.num_classes)
        else:
            self.classifier = nn.Conv2d(self.num_features, self.num_classes,
                                        kernel_size=1, stride=1)  # 1x1 conv, change channels
        self.avg = nn.AdaptiveAvgPool2d(1)  # target output size 1x1

    def forward(self, x):
        x = self.features(x)
        if self.fc_or_fcn == 'fc':
            x = self.avg(x).view(-1, self.num_features)  # 1,n_fea
            x = self.classifier(x)
        else:
            x = self.classifier(x)  # conv to (7,7,num_classes)
            x = self.avg(x).view(-1, self.num_classes)  # 1,n_cls
        return x


@register
def ft_resnet(mode: str = 'resnet50',
              fc_or_fcn: str = 'fc',
              num_classes: int = 10,
              pretrained: bool = True) -> nn.Module:
    """Finetune resnet.
    """
    return FT_Resnet(mode, fc_or_fcn, num_classes, pretrained)
