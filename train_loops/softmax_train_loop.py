import pytorch_lightning as pl
import torch
import torchmetrics


class SoftmaxTrainLoop(pl.LightningModule):

    def __init__(self, backbone, backbone_emb_size=None, num_classes=None):
        # The optional params are here to have a compatible api as the other train loops
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
        return self.get_embs(x)
    
    def get_embs(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
    def general_step(self, batch, phase, acc):
        x, y = batch
        y_hat = self.backbone(x)

        step_loss = self.loss(y_hat, y)
        step_acc = acc(y_hat, y)
        
        self.log(f'{phase}_loss', step_loss, on_epoch=True, on_step=False)
        self.log(f'{phase}_acc', step_acc, on_epoch=True, on_step=False)
        
        return step_loss
        
    def training_step(self, batch, batch_idx):
        return self.general_step(batch, 'train', self.train_acc)
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, 'val', self.val_acc)