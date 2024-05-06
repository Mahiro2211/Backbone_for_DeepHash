import torch
from torchvision import models
import torch.nn as nn

class ResHashLayer(nn.Module):
    def __init__(self, hash_bit):
        super(ResHashLayer, self).__init__()
        self.head = nn.Linear(1000, 1024)
        self.FcLayer = nn.Sequential(
            nn.Dropout(),
            self.head,
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit),
            # nn.BatchNorm1d(hash_bit,momentum=0.1)
        )
    def forward(self, feature):

        hash_re = self.FcLayer(feature)

        return hash_re


class ResNet50_(nn.Module):
    def __init__(self,bit):
        super(ResNet50_,self).__init__()
        self.backbone = models.resnet50(True)
        self.neck = ResHashLayer(bit)
    def forward(self,x):
        return self.neck(self.backbone(x))

# net = ResNet50_(16)
#
# print(net)
# a = torch.randn((16,3,224,224))
#
# b = net(a)
# print(b.shape)
