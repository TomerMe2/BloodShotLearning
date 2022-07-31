import torch
import pytorch_lightning as pl
import torchmetrics
from pytorch_metric_learning import losses
from torchvision.transforms.functional import rotate

class TrainingLoopMetricLearningConsistentLossArcfaceLoss(pl.LightningModule):

  def __init__(self, backbone, num_classes, consistent_loss_multiplier=20):
    super().__init__()
    self.backbone = backbone
    self.consistent_loss_multiplier = consistent_loss_multiplier
    self.loss = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=576)
    self.mse_loss = torch.nn.MSELoss()
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
    step_metrics_loss = self.loss(embs, y)
    self.log(f'{stage}_metric_loss', step_metrics_loss, on_step=False, on_epoch=True)

    x_for_consistent, embs_for_consistent = [], []
    for single_x, single_emb in zip(x, embs):
        for i in range(10):
            single_x = rotate(single_x, torch.rand(1).item() * 360)
            x_for_consistent.append(single_x)
            embs_for_consistent.append(single_emb)
    
    x_for_consistent, embs_for_consistent = torch.stack(x_for_consistent), torch.stack(embs_for_consistent)
    embs_of_rotated = self.get_embs(x_for_consistent)
    
    step_consistent_loss = self.mse_loss(embs_for_consistent, embs_of_rotated)
    self.log(f'{stage}_consistent_loss', step_consistent_loss, on_step=False, on_epoch=True)

    step_loss = step_metrics_loss + self.consistent_loss_multiplier * step_consistent_loss
    self.log(f'{stage}_loss', step_loss, on_step=False, on_epoch=True)

    return step_loss

  def training_step(self, train_batch, batch_idx):
    return self.general_step(train_batch, 'train', self.train_acc)
    
  def validation_step(self, val_batch, batch_idx):
    return self.general_step(val_batch, 'val', self.val_acc)  
