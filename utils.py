import torch

from train_loops.softmax_train_loop import SoftmaxTrainLoop
from train_loops.aam_softmax_train_loop import AAMSoftmaxTrainLoop
from train_loops.aam_softmax_consistency_loss_train_loop import AAMSoftmaxConsistencyLossTrainLoop


def dataset_split(dataset):
    train_part, val_part, test_part = 0.7, 0.1, 0.2

    amount_of_samples_train = int(train_part * len(dataset))
    amount_of_samples_val = int(val_part * len(dataset))
    amount_of_samples_test = len(dataset) - amount_of_samples_train - amount_of_samples_val
    
    print(amount_of_samples_train, amount_of_samples_val, amount_of_samples_test)

    train, val, test = torch.utils.data.random_split(dataset,
                                                     [amount_of_samples_train, amount_of_samples_val, amount_of_samples_test],
                                                     generator=torch.Generator().manual_seed(42))

    return train, val, test


def get_train_loop_cls_ref(train_loop_name):
    if train_loop_name == 'softmax':
        return SoftmaxTrainLoop
    elif train_loop_name == 'AAMSoftmax':
        return AAMSoftmaxTrainLoop
    elif train_loop_name == 'AAMSoftmaxConsistency':
        return AAMSoftmaxConsistencyLossTrainLoop
    
    raise Exception('Unknown train loop requested in command line arguemnts')
