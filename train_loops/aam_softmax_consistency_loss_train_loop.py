import pytorch_lightning as pl
import torch
import torchmetrics

from losses.sub_center_arc_face_loss_with_logits_out import SubCenterArcFaceLossWithLogitsOut
from losses.consistency_loss import ConsistencyLoss

class AAMSoftmaxConsistencyLossTrainLoop(pl.LightningModule):

    def __init__(self, backbone, backbone_emb_size, num_classes, consistency_loss_mult=1):
        super().__init__()
        
        self.backbone = backbone
        self.metric_loss = SubCenterArcFaceLossWithLogitsOut(num_classes=num_classes,
                                                             embedding_size=backbone_emb_size, sub_centers=1)
        self.consistency_loss = ConsistencyLoss(number_of_consistency_check_per_data_point=1)
        self.consistency_loss_mult = consistency_loss_mult
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_output_with_clf_head(self, x):
        x = self.get_embs(x)
        _, x = self.metric_loss(x, torch.zeros(x.shape[0], dtype=int))
        return x
    
    def get_embs(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
    def get_single_emb(self, img):
        img = torch.unsqueeze(img, dim=0)
        emb = self.get_embs(img)
        return emb[0]
    
    def general_step(self, batch, phase, acc):
        x, y = batch
        embs = self.get_embs(x)

        step_metric_loss, y_hat = self.metric_loss(embs, y)
        step_consistency_loss = self.consistency_loss(x, embs, self.get_embs)
        step_loss = step_metric_loss + self.consistency_loss_mult * step_consistency_loss
        
        step_acc = acc(y_hat, y)
        
        self.log(f'{phase}_metric_loss', step_metric_loss, on_epoch=True, on_step=False)
        self.log(f'{phase}_consistency_loss', step_consistency_loss, on_epoch=True, on_step=False)
        self.log(f'{phase}_loss', step_loss, on_epoch=True, on_step=False)
        self.log(f'{phase}_acc', step_acc, on_epoch=True, on_step=False)
        
        return step_loss
        
    def training_step(self, batch, batch_idx):
        return self.general_step(batch, 'train', self.train_acc)
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, 'val', self.val_acc)