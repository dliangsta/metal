import torch.nn as nn
from torchvision import models
from metal.contrib.modules.module import Module

class Densenet(Module):

  def __init__(self, pretrained=True, freeze=False, **kwargs):
    super().__init__()
    self.model = models.densenet201(pretrained=pretrained)
    self.model.classifier = nn.Sequential()
    if freeze:
      self.freeze()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    model = models.densenet201(pretrained=True)
    last_layer_output_size = int(model.classifier.weight.size()[1])
    del model
    return last_layer_output_size