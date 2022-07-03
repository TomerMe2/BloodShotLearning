import pytorch_lightning as pl

class TrainingLoop(pl.LightningModule):
  def __init__(self, backbone):
    super().__init__()
    self.backbone = backbone
    self.loss = torch.nn.CrossEntropyLoss()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters())
    return optimizer

  def forward(self, x):
    return self.backbone(x)
  
  def general_step(self, batch, stage):
    x, y = batch
    y_hat = self.backbone(x)
    step_loss = self.loss(y_hat, y)
    print('alo alo')
    self.log(f'{stage}_loss', step_loss)
    return step_loss

  def training_step(self, train_batch, batch_idx):
    return self.general_step(train_batch, 'train')

  def validation_step(self, val_batch, batch_idx):
    return self.general_step(val_batch, 'val')  
