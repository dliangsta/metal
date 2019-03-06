import pretrainedmodels
import torch.nn as nn
from metal.contrib.modules.module import Module

class PretrainedModel(Module):

  def __init__(self, pretrained=True, freeze=False, model_name="se_resnext101_32x4d", **kwargs):
    super().__init__()
    self.model_name = model_name
    self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
    self.model.last_linear = nn.Sequential()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    return 2048