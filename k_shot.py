import argparse
import os
from pprint import pprint
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from tqdm import tqdm

from datasets.c_nmc_leukemia_dataset import CNmcLeukemiaTrainingDataset, CNmcLeukemiaTestingDataset
from utils import get_train_loop_cls_ref

"""
This file will run a k-shot evaluation of a trained model.
Example for a command:
python k_shot.py --k 10 --train-loop AAMSoftmaxConsistency --checkpoint-path aam_softmax_consistency_model.ckpt
"""


def gather_k_examples(k, train_dataset, rng=None):
    """
    Gather k examples from each class from train_dataset using rng random generator.
    Useful for gathering k images in the memorizing phase.
    Returns a dict that maps from label names into list of indices in the dataset.
    """
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    samples_idxs_for_memorizing = {}  # dict of lbl_name -> list of indices in dataset
    for lbl in train_dataset.lbls_sorted:
        idxs_of_samples_with_lbl = [idx_of_sample for idx_of_sample, lbl_of_sample in enumerate(train_dataset.imgs_lbls) if
                                                                                      train_dataset.lbl_encoder_inverse[lbl_of_sample] == lbl]
        samples_idxs_for_memorizing[lbl] = rng.choice(idxs_of_samples_with_lbl, k)

    return samples_idxs_for_memorizing


def generated_mean_embedding_per_class(samples_idxs_for_memorizing, train_dataset, model):
    """
    samples_idxs_for_memorizing should be a dict that maps from label names into list of indices in the dataset.
    Returns a tuple in which the first item is a matrix in size of (n_classes, emb_size)
    and the second item is a np array the dictates the labels of each embedding in the first item.
    """
    
    representative_embs, rep_embs_lbls = [], []
    for lbl, idxs in samples_idxs_for_memorizing.items():
        samples = torch.stack([train_dataset[idx][0] for idx in idxs]).cuda()
        
        with torch.no_grad():
            embs = model.get_embs(samples)
        
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
            embs = model.get_embs(x).cpu().numpy()
        
        idxs_of_lbls = cosine_similarity(embs, representative_embs).argmax(axis=1)
        pred_lbls = rep_embs_lbls[idxs_of_lbls]
        predictions.extend(pred_lbls.tolist())
        true_lbls.extend(y.numpy().tolist())

    predictions = np.array(predictions)
    true_lbls = np.array([test_dataset.lbl_encoder_inverse[lbl] for lbl in true_lbls])

    metrics_dct = classification_report(true_lbls, predictions, output_dict=True)
    # we don't care about macro avg, only about weighted avg
    del metrics_dct['macro avg']
    
    return metrics_dct


def reduce_multiple_metrics_dcts(metrics_dcts):
    new_metrics_dct = {}
    
    # all the dcts have the same entries
    for key in metrics_dcts[0].keys():
        
        if key == 'accuracy':
            # accuracy is not a dict, it has a single value
            acc_values = [metric_dct[key] for metric_dct in metrics_dcts]

            new_metrics_dct[key] = {}
            new_metrics_dct[key]['mean'] = np.mean(acc_values)
            new_metrics_dct[key]['std'] = np.std(acc_values)
            
        else:
            new_metrics_dct[key] = {}
        
            for metric_nm in metrics_dcts[0][key].keys():
                metric_values = [metric_dct[key][metric_nm] for metric_dct in metrics_dcts]
                
                new_metrics_dct[key][metric_nm] = {}
                new_metrics_dct[key][metric_nm]['mean'] = np.mean(metric_values)
                new_metrics_dct[key][metric_nm]['std'] = np.std(metric_values)

    return new_metrics_dct


def k_shot(k, model, train_dataset, test_dataset, num_of_evals, seed=42):
    # k images should fit in the GPU.
    rng = np.random.default_rng(seed)
    
    metrics_dcts = []
    for eval_idx in tqdm(range(num_of_evals)):
        samples_idxs_for_memorizing = gather_k_examples(k, train_dataset, rng)
        representative_embs, rep_embs_lbls = generated_mean_embedding_per_class(samples_idxs_for_memorizing,
                                                                                train_dataset, model)
        metrics_dct = classify_with_representative_emb(test_dataset, model, representative_embs, rep_embs_lbls)
        metrics_dcts.append(metrics_dct)
    
    final_metric_dct = reduce_multiple_metrics_dcts(metrics_dcts)
    return final_metric_dct


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int,
                        help='Number of memorizing image per evaluation')
    parser.add_argument('--train-loop', type=str, 
                        help='The training loop that the model was trained with. Can be softmax or AAMSoftmax or AAMSoftmaxConsistency')
    parser.add_argument('--checkpoint-path', type=str,
                        help='Path to the ckpt file that contains the weights')
    parser.add_argument('--all-dataset-path', 
                        default='../C-NMC-Leukemia/C-NMC_Leukemia',
                        type=str, help='Path to the unzipped ALL dataset')
    parser.add_argument('--number-of-evals', default=100, type=int,
                        help='Number of times that we pick memorizing images and evaluate the model using them')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loop_cls_ref = get_train_loop_cls_ref(args.train_loop)
    
    model = train_loop_cls_ref.load_from_checkpoint(args.checkpoint_path)
    model = model.cuda()
    model = model.eval()
    
    memorizing_dataset = CNmcLeukemiaTrainingDataset(os.path.join(args.all_dataset_path, 'C-NMC_training_data', 'fold_0'))
    testing_dataset = CNmcLeukemiaTestingDataset(os.path.join(args.all_dataset_path, 'C-NMC_test_prelim_phase_data'))

    pprint(k_shot(args.k, model, memorizing_dataset, testing_dataset, args.number_of_evals))
