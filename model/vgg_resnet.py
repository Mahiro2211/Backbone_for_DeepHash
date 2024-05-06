import torchvision.models as model
import torch.nn as nn
class VGG16(nn.Module):
    def __init__(self,bit):
        super(VGG16,self).__init__()
        self.backbone = model.vgg16(True)
        self.backbone.classifier = self.backbone.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096,2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048,bit),
            nn.BatchNorm1d(bit,momentum=0.1)
        )
    def forward(self,x):
        return self.hash_layer(self.backbone(x))

net = VGG16(16)

print(net)

