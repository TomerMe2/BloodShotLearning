import torch
import pytorch_lightning as pl
import torchvision

from aml_dataset import AMLDataset
from training_loop import TrainingLoop

train_part, val_part, test_part = 0.7, 0.1, 0.2
INPUT_FOLDER = '../AML_Cytomorphology/AML_Cytomorphology'

if __name__ == '__main__':
    aml_dataset = AMLDataset(INPUT_FOLDER) 

    amount_of_samples_train = int(train_part * len(aml_dataset))
    amount_of_samples_val = int(val_part * len(aml_dataset))
    amount_of_samples_test = len(aml_dataset) - amount_of_samples_train - amount_of_samples_val

    aml_train, aml_val, aml_test = torch.utils.data.random_split(aml_dataset,
                                                                [amount_of_samples_train, amount_of_samples_val, amount_of_samples_test],
                                                                generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(aml_train, batch_size=128, num_workers=2)
    val_loader = torch.utils.data.DataLoader(aml_val, batch_size=128, num_workers=2)

    backbone = torchvision.models.efficientnet_b0(pre_trained=False, num_classes=aml_dataset.num_classes)
    training_loop = TrainingLoop(backbone)

    trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0)
    trainer.fit(training_loop, train_loader, val_loader)

