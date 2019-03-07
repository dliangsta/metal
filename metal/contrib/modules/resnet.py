import torch.nn as nn
from torchvision import models

class Resnet(nn.Module):

  def __init__(self, pretrained=True, freeze=False, **kwargs):
    super().__init__()
    self.model = models.resnet152(pretrained=pretrained)
    self.model.fc = nn.Sequential()
    if freeze:
      self.freeze()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    return 2048