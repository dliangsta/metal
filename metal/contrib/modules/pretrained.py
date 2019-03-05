import pretrainedmodels
import torch.nn as nn
from metal.contrib.modules.module import Module

class PretrainedModel(Module):

  MODEL_NAME = 'inceptionresnetv2'

  def __init__(self, pretrained=True, freeze=False, **kwargs):
    super().__init__()
    self.model = pretrainedmodels.__dict__[PretrainedModel.MODEL_NAME](num_classes=1000, pretrained='imagenet')
    self.model.last_linear = nn.Sequential()
    print(self.model)

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    model = pretrainedmodels.__dict__[PretrainedModel.MODEL_NAME](num_classes=1000, pretrained='imagenet')
    last_layer_output_size = int(model.last_linear.weight.size()[1])
    del model
    return last_layer_output_size