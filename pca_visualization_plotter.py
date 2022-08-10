import argparse
from sklearn.decomposition import PCA
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from datasets.c_nmc_leukemia_dataset import CNmcLeukemiaTrainingDataset
from utils import get_train_loop_cls_ref

"""
This file generate a plot of embeddings after dimensionality reduction to 2D using PCA.
Example for a command:
python k_vs_accuracy_plotter.py --train-loop AAMSoftmaxConsistency --checkpoint-path aam_softmax_consistency_model.ckpt
"""


def choose_indices(number_of_examples, dataset, seed=42):
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(dataset), number_of_examples, replace=False)

    return chosen_indices


def indices_into_embeddings(chosen_indices, dataset, model):
    embs, lbls = [], []
    for idx_of_img in chosen_indices:
        img, lbl = dataset[idx_of_img]
        
        with torch.no_grad():
            emb = model.get_embs(torch.unsqueeze(img, dim=0))[0]
        
        embs.append(emb.numpy())
        lbls.append(lbl)
    
    return np.stack(embs), np.array(lbls)


def pca_and_plot(chosen_embs, lbls, lbl_encoder_inverse, out_path):
    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(chosen_embs)
    
    for lbl_idx in np.unique(lbls):
        lbl_mask = lbls == lbl_idx
        embs_2d_of_lbl = embs_2d[lbl_mask]
        
        lbl_name = lbl_encoder_inverse[lbl_idx]
        lbl_name = 'cancer' if lbl_name == 'all' else 'normal'
            
        plt.scatter(embs_2d_of_lbl[:, 0], embs_2d_of_lbl[:, 1], label=lbl_name)
    
    plt.title('Visualization of Embeddings After PCA Transformation')
    plt.legend()
    plt.show()
    plt.savefig(out_path)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-loop', type=str, 
                        help='The training loop that the model was trained with. Can be softmax or AAMSoftmax or AAMSoftmaxConsistency')
    parser.add_argument('--checkpoint-path', type=str,
                        help='Path to the ckpt file that contains the weights')
    parser.add_argument('--all-dataset-path', 
                        default='../C-NMC-Leukemia/C-NMC_Leukemia',
                        type=str, help='Path to the unzipped ALL dataset')
    parser.add_argument('--number-of-points-to-plot', default=400, type=int,
                        help="Number of images to plot the PCA of their's embedding in the 2D plot")
    parser.add_argument('--plot-output-path', default='pca_visualization.pdf', type=str,
                        help='Output path for the plot. Should be a pdf format.')

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loop_cls_ref = get_train_loop_cls_ref(args.train_loop)
    
    model = train_loop_cls_ref.load_from_checkpoint(args.checkpoint_path)
    model = model.eval()
    
    dataset = CNmcLeukemiaTrainingDataset(os.path.join(args.all_dataset_path, 'C-NMC_training_data', 'fold_0'))
    
    chosen_indices = choose_indices(400, dataset)
    embs, lbls = indices_into_embeddings(chosen_indices, dataset, model)
    pca_and_plot(embs, lbls, dataset.lbl_encoder_inverse, args.plot_output_path)