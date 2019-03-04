import torch
import torch.nn as nn
from torchvision import models
import imp

class DensenetDDSM(nn.Module):

  def __init__(self, freeze=False, **kwargs):
    super().__init__()
    MainModel = imp.load_source("MainModel", "weights/weights.py")
    self.model = torch.load("weights/weights.best.pytorch.hdf5")
    self.model._modules["dense_1"] = torch.nn.Sequential()
    if freeze:
      self.freeze()

  def forward(self, x):
    return self.model(x)

  @staticmethod
  def last_layer_output_size():
    MainModel = imp.load_source("MainModel", "weights/weights.py")
    model = torch.load("weights/weights.best.pytorch.hdf5")
    last_layer_output_size = int(model._modules["dense_1"].weight.size()[1])
    del model
    return last_layer_output_size