import torch.nn as nn
from torchvision.models import resnet
import torch
import numpy as np
device = 'cuda' #if torch.cuda.is_initialized() else 'cpu'
torch.device(device)


class MLP(nn.Module):
    def __init__(self, inputsize, h1size, h2size, outputsize):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(inputsize, h1size),
                                   nn.ReLU(),
                                   nn.Linear(h1size, h2size),
                                   nn.ReLU(),
                                   nn.Linear(h2size, h2size),
                                   nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h2size + 1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, h2size),
                                  nn.ReLU(),
                                  nn.Linear(h2size, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, outputsize))
        self.backbone = resnet.resnet18(pretrained=True, progress=True)
        self.backbone.layer4 = nn.Sequential()
        self.reduce_chn = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())


    def forward(self, feature, image):
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.reduce_chn(x)
        b, _, _, _ = x.shape
        x = x.view((b, -1))
        f = self.model(feature)
        out = torch.cat([f, x], axis=-1)

        return self.head(out)

class MLP_without_CNN(nn.Module):
    def __init__(self, inputsize, h1size, h2size, outputsize):
        super(MLP_without_CNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(inputsize, h1size),
                                   nn.ReLU(),
                                   nn.Linear(h1size, h2size),
                                   nn.ReLU(),
                                   nn.Linear(h2size, h2size),
                                   nn.ReLU())
        self.head = nn.Sequential(nn.Linear(inputsize + 3*28*28, h1size),
                                  nn.ReLU(),
                                  nn.Linear(h1size, h2size),
                                  nn.ReLU(),
                                  nn.Linear(h2size, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                 
                                  nn.ReLU(),
                                  nn.Linear(32, outputsize))
        # self.backbone = resnet.resnet18(pretrained=True, progress=True)
        # self.backbone.layer4 = nn.Sequential()
        # self.reduce_chn = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1),
        #                                 nn.BatchNorm2d(16),
        #                                 nn.ReLU())


    def forward(self, feature, image):
#         print(image.size())
        x=image.view(-1,image.size()[1]*image.size()[2]*image.size()[3])
        
#         f = self.model(feature)
        x_=x
#         for _ in range(f.size()[0]//x.size()[0]-1):
#             x_=torch.cat([x_,x],0)
#         print(f.size(),x_.size())
#         exit(0)
        out = torch.cat([feature, x_], axis=-1)
        if torch.isnan(out).any():
            print('nan!')
            exit(1)
        preds=self.head(out)
#         print(out[63],preds[63])
        
        return preds

