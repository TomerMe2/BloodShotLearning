import argparse
import os
import matplotlib.pyplot as plt

from k_shot import k_shot
from datasets.c_nmc_leukemia_dataset import CNmcLeukemiaTrainingDataset, CNmcLeukemiaTestingDataset
from utils import get_train_loop_cls_ref


KS = [1, 2, 3, 5, 10, 15, 20, 30, 50]

"""
This file generate a plot of k (number of memorizing images) vs accuracy.
Example for a command:
python k_vs_accuracy_plotter.py --train-loop AAMSoftmaxConsistency --checkpoint-path aam_softmax_consistency_model.ckpt
"""

def make_plot(model, memorizing_dataset, testing_dataset, num_of_evals_per_k, ks, out_path):
    
    accs = []
    for k in ks:
        res = k_shot(k, model, memorizing_dataset, testing_dataset, num_of_evals=num_of_evals_per_k)
        acc = res['accuracy']['mean']
        
        print(k, acc)
        accs.append(acc)
    
    plt.plot(ks, accs, marker='o')
    plt.title('Number of Memorizing Images Vs Accuracy')
    plt.xlabel('Number of Memorizing Images Per Class')
    plt.ylabel('Accuracy')
    
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
    parser.add_argument('--number-of-evals', default=20, type=int,
                        help='Number of times that we pick memorizing images and evaluate the model using them for each number of memorizing images that we evaluate')
    parser.add_argument('--plot-output-path', default='k_vs_accuracy.pdf', type=str,
                        help='Output path for the plot. Should be a pdf format.')

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

    make_plot(model, memorizing_dataset, testing_dataset, args.number_of_evals, KS, args.plot_output_path)

