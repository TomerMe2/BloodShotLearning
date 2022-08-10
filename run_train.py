import argparse
import torch
import pytorch_lightning as pl
import torchvision

from datasets.aml_dataset import AMLDataset
from utils import dataset_split, get_train_loop_cls_ref

EFFICIENT_NET_EMB_SIZE = 1280


"""
This file will train a neural network using the specified training loop.
Example for a command:
python run_train.py --train-loop AAMSoftmaxConsistency --experiment-name aam_softmax_consistency_run
"""


def train(dataset_input_folder, train_loop, experiment_name, batch_size, num_workers):
    aml_dataset = AMLDataset(dataset_input_folder)
    aml_train, aml_val, _ = dataset_split(aml_dataset)
    
    pl.seed_everything(123)
    
    train_loader = torch.utils.data.DataLoader(aml_train, batch_size=batch_size, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(aml_val, batch_size=batch_size, num_workers=num_workers)

    backbone = torchvision.models.efficientnet_b0()
    training_loop = train_loop(backbone, EFFICIENT_NET_EMB_SIZE, aml_dataset.num_classes)
    
    logger = pl.loggers.CSVLogger("logs", name=experiment_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=-1)
    
    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         gpus=1, max_epochs=6, num_sanity_val_steps=0, logger=logger)
    trainer.fit(training_loop, train_loader, val_loader)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-loop', type=str, 
                        help='Can be softmax or AAMSoftmax or AAMSoftmaxConsistency')
    parser.add_argument('--experiment-name', type=str,
                        help='The name that your experiment will be saved under in the logs directory')
    parser.add_argument('--aml-dataset-path', 
                        default='../AML-Cytomorphology/AML-Cytomorphology',
                        type=str, help='Path to the unzipped AML dataset')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=22, type=int)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loop_cls_ref = get_train_loop_cls_ref(args.train_loop)
    
    train(args.aml_dataset_path, train_loop_cls_ref, args.experiment_name,
          args.batch_size, args.num_workers)
    