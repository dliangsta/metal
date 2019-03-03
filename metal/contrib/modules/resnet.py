import torch.nn as nn
from torchvision import models

class Resnet18(nn.Module):

  def __init__(self, pretrained=True, **kwargs):
    super().__init__()
    self.model = models.resnet18(pretrained=pretrained)
    self.model.fc = nn.Sequential()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    model = models.resnet18(pretrained=True)
    last_layer_output_size = int(model.fc.weight.size()[1])
    del model
    return last_layer_output_size