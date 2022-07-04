import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

from datasets.c_nmc_leukemia_dataset import CNmcLeukemiaTrainingDataset, CNmcLeukemiaTestingDataset
from training_loop import TrainingLoop


# CHECKPOINT_PATH = 'logs/init/version_12/checkpoints/epoch=8-step=3609.ckpt'
CHECKPOINT_PATH = 'logs/init/version_14/checkpoints/epoch=0-step=401.ckpt'
TRAIN_DATASET_PATH = '../C-NMC-Leukemia/C-NMC_Leukemia/C-NMC_training_data/fold_0'
TEST_DATASET_PATH = '../C-NMC-Leukemia/C-NMC_Leukemia/C-NMC_test_prelim_phase_data'

# TODO: make it nicer
def get_embs_in_efficient_net(model, x):
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x

def gather_k_examples(k, train_dataset):
    rng = np.random.default_rng(42)
    
    samples_idxs_for_memorizing = {}  # dict of lbl_name -> list of indices in dataset
    for lbl in train_dataset.lbls_sorted:
        idxs_of_samples_with_lbl = [idx_of_sample for idx_of_sample, lbl_of_sample in enumerate(train_dataset.imgs_lbls) if
                                                                                      train_dataset.lbl_encoder_inverse[lbl_of_sample] == lbl]
        samples_idxs_for_memorizing[lbl] = rng.choice(idxs_of_samples_with_lbl, k)

    return samples_idxs_for_memorizing


def generate_embeddings_with_mean(samples_idxs_for_memorizing, train_dataset, model):
    
    representative_embs, rep_embs_lbls = [], []
    for lbl, idxs in samples_idxs_for_memorizing.items():
        samples = torch.stack([train_dataset[idx][0] for idx in idxs]).cuda()
        
        with torch.no_grad():
            embs = get_embs_in_efficient_net(model, samples)
        
        representative_embs.append(embs.mean(dim=0).cpu())
        rep_embs_lbls.append(lbl)

    return torch.stack(representative_embs), np.array(rep_embs_lbls)


def classify_with_representative_emb(test_dataset, model, representative_embs, rep_embs_lbls):
    representative_embs = representative_embs.numpy()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=22)

    predictions, true_lbls = [], []
    for x, y in test_loader:
        x = x.cuda()

        with torch.no_grad():
            embs = get_embs_in_efficient_net(model, x).cpu().numpy()
        
        idxs_of_lbls = cosine_similarity(embs, representative_embs).argmax(axis=1)
        pred_lbls = rep_embs_lbls[idxs_of_lbls]
        predictions.extend(pred_lbls.tolist())
        true_lbls.extend(y.numpy().tolist())

    predictions = np.array(predictions)
    true_lbls = np.array([test_dataset.lbl_encoder_inverse[lbl] for lbl in true_lbls])

    print(classification_report(true_lbls, predictions))

    print('predict all on everything')
    predictions = np.array(['all'] * len(predictions))
    print(classification_report(true_lbls, predictions))



        

def k_shot(k, model, train_dataset, test_dataset):
    # k needs to be lower than batch size. k images should fit in the GPU.
    samples_idxs_for_memorizing = gather_k_examples(k, train_dataset)
    representative_embs, rep_embs_lbls = generate_embeddings_with_mean(samples_idxs_for_memorizing, train_dataset, model)
    classify_with_representative_emb(test_dataset, model, representative_embs, rep_embs_lbls)


if __name__ == '__main__':
    model = TrainingLoop.load_from_checkpoint(CHECKPOINT_PATH)
    model = model.backbone.cuda()
    model = model.eval()

    memorizing_dataset = CNmcLeukemiaTrainingDataset(TRAIN_DATASET_PATH)
    testing_dataset = CNmcLeukemiaTestingDataset(TEST_DATASET_PATH)

    k_shot(10, model, memorizing_dataset, testing_dataset)
   


    


