import torch.nn as nn
from torchvision import models

class Densenet121(nn.Module):

  def __init__(self, pretrained=True, **kwargs):
    super().__init__()
    self.model = models.densenet121(pretrained=pretrained)
    self.model.classifier = nn.Sequential()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    model = models.densenet121(pretrained=True)
    last_layer_output_size = int(model.classifier.weight.size()[1])
    del model
    return last_layer_output_size