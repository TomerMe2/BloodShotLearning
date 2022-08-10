import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report

from datasets.aml_dataset import AMLDataset
from utils import dataset_split, get_train_loop_cls_ref

#CHECKPOINT_PATH = 'logs/subcenter_1_arcface_mobilenet_v3_consistency_mult_1_with_logits_with_less_augs/version_0/checkpoints/epoch=5-step=2406_backup.ckpt'
CHECKPOINT_PATH = 'logs/arcface_again_effnet_consistency_loss/version_11/checkpoints/epoch=5-step=2406.ckpt'

""""
This file prints the classification report on the test of the AML dataset
Example for a command:
python test_normal_classification.py --train-loop AAMSoftmaxConsistency --checkpoint-path aam_softmax_consistency_model.ckpt
"""



def classify_test(model, test_dataset, num_classes, lbl_encoder_inverse,
                  batch_size, num_workers):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, num_workers=num_workers)

    y_preds, y_trues = [], []
    for x, y in test_loader: 
        
        with torch.no_grad():
            y_pred = model.get_output_with_clf_head(x.cuda())

        y_pred = y_pred.argmax(dim=1).cpu().numpy()
        y_preds.append(y_pred)
        y_trues.append(y.cpu().numpy())

    y_preds = np.hstack(y_preds)
    y_trues = np.hstack(y_trues)
    
    target_names = [lbl_encoder_inverse[lbl] for lbl in range(num_classes)]
    print(classification_report(y_trues, y_preds, target_names=target_names))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-loop', type=str, 
                        help='The training loop that the model was trained with. Can be softmax or AAMSoftmax or AAMSoftmaxConsistency')
    parser.add_argument('--checkpoint-path', type=str,
                        help='Path to the ckpt file that contains the weights')
    parser.add_argument('--aml-dataset-path', 
                        default='../AML-Cytomorphology/AML-Cytomorphology',
                        type=str, help='Path to the unzipped AML dataset')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=22, type=int)
    return parser


if __name__ == '__main__':
    #model = TrainingLoop.load_from_checkpoint(CHECKPOINT_PATH)
    #model = TrainingLoopMetricLearning.load_from_checkpoint(CHECKPOINT_PATH)
    #model = TrainingLoopMetricLearningConsistentLoss.load_from_checkpoint(CHECKPOINT_PATH)
    # model = TrainingLoopAAMSoftmax.load_from_checkpoint(CHECKPOINT_PATH)
    #model = TrainingLoopAAMSoftmaxWithLogits.load_from_checkpoint(CHECKPOINT_PATH)
    #model = FallbackTrainingLoop.load_from_checkpoint(CHECKPOINT_PATH)
    #model = TrainingLoopMetricLearningConsistentLossArcfaceLoss.load_from_checkpoint(CHECKPOINT_PATH)
    #model = TrainingLoopSubCenterWithLogitsOutAndConsistent.load_from_checkpoint(CHECKPOINT_PATH)
    # model = ArcfaceTrainloopAgainConsistencyLoss.load_from_checkpoint(CHECKPOINT_PATH)
    # model = model.cuda()
    # model = model.eval()
    
    # dataset = AMLDataset('../AML-Cytomorphology/AML-Cytomorphology') 
    # _, _, test_dataset = dataset_split(dataset)

    # classify_test(model, test_dataset, dataset.num_classes, dataset.lbl_encoder_inverse)
    
    
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loop_cls_ref = get_train_loop_cls_ref(args.train_loop)
    
    model = train_loop_cls_ref.load_from_checkpoint(args.checkpoint_path)
    model = model.cuda()
    model = model.eval()
    
    dataset = AMLDataset(args.aml_dataset_path)
    _, _, test_dataset = dataset_split(dataset)

    classify_test(model, test_dataset, dataset.num_classes,
                  dataset.lbl_encoder_inverse, args.batch_size, args.num_workers)
    
    