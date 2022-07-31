import torch
import pytorch_lightning as pl
import torchmetrics
from aam_softmax_with_logits_out import AAMSoftmaxWithLogitsOut

class TrainingLoopAAMSoftmaxWithLogits(pl.LightningModule):

  def __init__(self, backbone, num_classes, sub_centers=3):
    super().__init__()
    self.backbone = backbone
    self.loss = AAMSoftmaxWithLogitsOut(num_classes=num_classes, embedding_size=576)
    self.train_acc = torchmetrics.Accuracy()
    self.val_acc = torchmetrics.Accuracy()
    self.save_hyperparameters()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    return optimizer
  
  def get_embs(self, x):
    x = self.backbone.features(x)
    x = self.backbone.avgpool(x)
    x = torch.flatten(x, 1)

    return x

  def forward(self, x):
    return self.backbone(x)
  
  def general_step(self, batch, stage, acc):
    x, y = batch
    embs = self.get_embs(x)
    step_loss, y_hat = self.loss(embs, y)
    acc(y_hat, y)

    self.log(f'{stage}_loss', step_loss, on_step=False, on_epoch=True)
    self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True)

    return step_loss

  def training_step(self, train_batch, batch_idx):
    return self.general_step(train_batch, 'train', self.train_acc)
    
  def validation_step(self, val_batch, batch_idx):
    return self.general_step(val_batch, 'val', self.val_acc)  
