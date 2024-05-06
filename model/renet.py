import torch
import torchvision.models as models
import torch.nn as nn
class HashLayer(nn.Module):
    def __init__(self, hash_bit):
        super(HashLayer, self).__init__()
        self.head = nn.Linear(2048, 1024)
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

class ResNet50(nn.Module):
    def __init__(self,bit):
        super(ResNet50,self).__init__()
        self.backbone = models.resnet50(True)
        self.backbone.fc = HashLayer(bit)

    def forward(self,x):
        return self.backbone(x)

# a = ResNet50(16)
# print(a)