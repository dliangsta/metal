import torch.nn as nn

class Module(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = None

  def freeze(self):
    assert self.model
    print("Freezing model!")
    for param in self.model.parameters():
      param.requires_grad = False

  @staticmethod
  def last_layer_output_size():
    raise NotImplementedError
    