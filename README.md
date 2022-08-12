# BloodShot: Few-shot Learning On Blood Cells
This repository contains the training code, evaluation code, trained model and report of BloodShot.

## Motivation
Labeling is an expensive process.
We wanted to create a pipeline that can classify images from an almost unseen dataset in an accuracy which is better than random guess.
Almost unseen dataset mean that the classifier have access only to a few labeled images from this dataset.
In the evaluation of this project we used 10 images per class.

## Datasets
The dataset that we can fully use is [A Single-cell Morphological Dataset of Leukocytes from AML Patients
and Non-malignant Controls](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958). (AML for short)<br/>
The dataset that we will have an access only to few images from its train dataset is [ALL Challenge dataset of ISBI 2019"](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223) (ALL-challenge for short).

## Method
In the train phase, train EfficientNetB0 on the AML dataset using AAMSoftmax and consistency loss.
In the memorize phase, gather k images per class from the training dataset of the ALL Challenge dataset and memorize the mean embedding of each class.
In the inference phase, preform a cosine similarity between the embedding of the incoming image and the embedding of each class. We will say that the class of the incoming image is the class of the closest mean embedding.

## Results

| Description | Accuracy on the ALL Dataset Using 10 Memorizing Images  |
|:-:|:-:|
| Softmax Loss   | 0.442±0.088 |
|  AAMSoftmax Loss | 0.590±0.029 |
| AAMSoftmax Loss & Concictency Loss  | 0.639±0.047 |

The base-rate (random guess) accuracy is 50% since the classifier has no access to the distribution of the test dataset because it gets only k examples from each class in the memorize phase.

## More Information & More Results
More information and more results can found in BloodShot's report.

## How To Run The Code
Download and unzip the AML dataset and the ALL-Challenge dataset.
The default unzipped data path is one directory above BloodShot's repo directory, but you can put the data in a different directory and specify the directory using command line arguments. 
This repo requires a GPU to run, please make sure that you install cuda properly before running this repo.
We used the docker image ```pytorch:1.11.0-cuda11.3-cudnn8-runtime``` as our environment, but any environment (pip or conda) with pytorch 1.11.0 should work. 
The packages that should be installed on top of the docker image that we used can be found in ```requirements.txt```

To train the embedder, please run:
```
python run_train.py --train-loop AAMSoftmaxConsistency --experiment-name aam_softmax_consistency_run
```

To evaluate the embedder, please run:
```
python k_shot.py --k 10 --train-loop AAMSoftmaxConsistency --checkpoint-path aam_softmax_consistency_model.ckpt
```
You can specify different checkpoints using the ```checkpoint-path``` argument.

## Acknowledgements
This project was done as a final project in Assaf Zaritsky's [Data Science in Cell Imaging course](https://assafzar.wixsite.com/dsci2022).
