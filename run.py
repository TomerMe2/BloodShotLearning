import torch
import pytorch_lightning as pl
import torchvision
from sklearn.metrics import classification_report
import numpy as np

from datasets.aml_dataset import AMLDataset
from training_loop import TrainingLoop
from training_loop_metrics_learning import TrainingLoopMetricLearning
from training_loop_metric_learning_consistent_loss import TrainingLoopMetricLearningConsistentLoss
from utils import dataset_split

INPUT_FOLDER = '../AML-Cytomorphology/AML-Cytomorphology'

if __name__ == '__main__':
    aml_dataset = AMLDataset(INPUT_FOLDER) 

    aml_train, aml_val, aml_test = dataset_split(aml_dataset)
    train_loader = torch.utils.data.DataLoader(aml_train, batch_size=32, num_workers=22)
    val_loader = torch.utils.data.DataLoader(aml_val, batch_size=32, num_workers=22)
    test_loader = torch.utils.data.DataLoader(aml_test, batch_size=32, num_workers=22)

    backbone = torchvision.models.mobilenet_v3_small(pre_trained=False, num_classes=aml_dataset.num_classes)
    # training_loop = TrainingLoop(backbone)
    # training_loop = TrainingLoopMetricLearning(backbone, aml_dataset.num_classes, sub_centers=1)
    training_loop = TrainingLoopMetricLearningConsistentLoss(backbone, aml_dataset.num_classes)
    logger = pl.loggers.CSVLogger("logs", name="arcface_loss_mobilenet_v3_consistent_loss_amp")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    # early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    #trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
    #                     gpus=1, max_epochs=100, num_sanity_val_steps=0, logger=logger)
    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         gpus=1, max_epochs=30, num_sanity_val_steps=0, logger=logger)
    trainer.fit(training_loop, train_loader, val_loader)


    net = training_loop.backbone
    net = net.cuda()
    net.eval()

    y_preds, y_trues = [], []
    for x, y in test_loader: 
        
        with torch.no_grad():
            y_pred = net(x.cuda())

        y_pred = y_pred.argmax(dim=1).cpu().numpy()
        y_preds.append(y_pred)
        y_trues.append(y.cpu().numpy())

    y_preds = np.hstack(y_preds)
    y_trues = np.hstack(y_trues)
    
    target_names = [aml_dataset.lbl_encoder_inverse[lbl] for lbl in range(aml_dataset.num_classes)]
    print(classification_report(y_trues, y_preds, target_names=target_names))