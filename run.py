import torch
import pytorch_lightning as pl
import torchvision
from sklearn.metrics import classification_report
import numpy as np

from aml_dataset import AMLDataset
from training_loop import TrainingLoop

train_part, val_part, test_part = 0.7, 0.1, 0.2
INPUT_FOLDER = '../AML-Cytomorphology/AML-Cytomorphology'

if __name__ == '__main__':
    aml_dataset = AMLDataset(INPUT_FOLDER) 

    amount_of_samples_train = int(train_part * len(aml_dataset))
    amount_of_samples_val = int(val_part * len(aml_dataset))
    amount_of_samples_test = len(aml_dataset) - amount_of_samples_train - amount_of_samples_val
    
    print(amount_of_samples_train, amount_of_samples_val, amount_of_samples_test)

    aml_train, aml_val, aml_test = torch.utils.data.random_split(aml_dataset,
                                                                [amount_of_samples_train, amount_of_samples_val, amount_of_samples_test],
                                                                generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(aml_train, batch_size=32, num_workers=22)
    val_loader = torch.utils.data.DataLoader(aml_val, batch_size=32, num_workers=22)

    backbone = torchvision.models.efficientnet_b0(pre_trained=False, num_classes=aml_dataset.num_classes)
    training_loop = TrainingLoop(backbone)

    logger = pl.loggers.CSVLogger("logs", name="init")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                         gpus=1, max_epochs=100, num_sanity_val_steps=0, logger=logger)

    trainer.fit(training_loop, train_loader, val_loader)

    test_loader = torch.utils.data.DataLoader(aml_test, batch_size=32, num_workers=22)

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