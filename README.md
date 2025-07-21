# CV_project
## Introduction
In this project we implement the main algorithm for plate detection which is formed by Yolov5 and pdlpr and then we compare it to a baseline method composed by traditional technique for plate area detection and CNN + CTC for plate character recognition

## Folder structure
- /dataset: here there are the images and the labels that are used for the training testing and validation, /cropped_images and /labels_pdlpr are used for the second part of character recognition in pdlpr and in the cnn + ctc method.
- /metrics_images: here there are the images generated during training by the models, they include the images for training and validation over epoch for character accuracy and full plate sequence accuracy
- /models: the saved models for CNNCTC that are then loaded for the testing phase
- /results: here there are txt files which contains the result metrics for the training and testing phase for each model
- /runs: results of the yolo models

## File structure
Since we implemented two different methods each one of them divided into two parts we decided to have separate files for training and evaluation (testing) for each method.

- globals.py: contains the constants used in this repository
- utils.py: useful functions and also the definition of the evaluator class  and the function that does plate detection using traditional techniques.
- network.py: definition of the NN for the PDLPR and the CNN CTC.
- data.py: the dataset class for CCPD is defined here

### Training
- train_baseline_cnnctc.py: The training code for the character recognition of the CNNCTC part for the baseline method, here we try different hyperparameters combination.
- train_pdlpr.py: The training code for the pdlpr character recognition part.
- train_yolo.py: The training code for Yolov5 license plate detection.

### Evaluation
These file are the testing for the single individual part
- test_baseline_cnnctc.py: testing for baseline CNNCTC method.
- test_pdlpr.py: testing for pdlpr
- test_baseline_traditional.py: testing for the plate detection using traditional techniques, this doesn't require training so it does not have the train file counterpart.
- test_yolo.py:

Then there are two files that implements two pipelines, that test the two methods (Yolov5 + pdlpr) and (traditional + cnnctc) by considering one image at a time, segmenting it and then inputting it directly to the character detection to make the final prediction.

- pipeline_testing.py: the pipeline of the method in the paper that loads our trained models and compute the accuracy
- pipeline_baseline_testing.py: the pipeline of the baseline method

