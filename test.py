import torch
import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, classCount, isTrained):

        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(kernelCount, classCount), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

# ckpt = torch.load('/home/eco0936_namnh/hn_lap/export_dl/m-30092022-104001.pth.tar')
# # print(ckpt.keys())
# # print(ckpt['state_dict'])
model = DenseNet121(classCount=14,isTrained=False)
# model.load_state_dict(ckpt['state_dict'])

# torch.save(model,'densenet121.pth')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./densenet121.pth',map_location=device)
print(model)
im = torch.zeros(1, 3, 224,224).to(device).float()
out = model(im)
print(out)