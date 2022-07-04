import torch
import pytorch_lightning as pl
import torchmetrics


class TrainingLoop(pl.LightningModule):

  def __init__(self, backbone):
    super().__init__()
    self.backbone = backbone
    self.loss = torch.nn.CrossEntropyLoss()
    self.train_acc = torchmetrics.Accuracy()
    self.val_acc = torchmetrics.Accuracy()
    self.save_hyperparameters()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    return optimizer

  def forward(self, x):
    return self.backbone(x)
  
  def general_step(self, batch, stage, acc):
    x, y = batch
    y_hat = self.backbone(x)
    step_loss = self.loss(y_hat, y)
    self.log(f'{stage}_loss', step_loss, on_step=False, on_epoch=True)

    acc(y_hat, y)
    self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True)
    
    return step_loss

  def training_step(self, train_batch, batch_idx):
    return self.general_step(train_batch, 'train', self.train_acc)
    
  def validation_step(self, val_batch, batch_idx):
    return self.general_step(val_batch, 'val', self.val_acc)  
