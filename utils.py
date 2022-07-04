import torch

def dataset_split(dataset):
    train_part, val_part, test_part = 0.7, 0.1, 0.2

    amount_of_samples_train = int(train_part * len(dataset))
    amount_of_samples_val = int(val_part * len(dataset))
    amount_of_samples_test = len(dataset) - amount_of_samples_train - amount_of_samples_val
    
    print(amount_of_samples_train, amount_of_samples_val, amount_of_samples_test)

    train, val, test = torch.utils.data.random_split(dataset,
                                                     [amount_of_samples_train, amount_of_samples_val, amount_of_samples_test],
                                                     generator=torch.Generator().manual_seed(42))

    return train,val, test