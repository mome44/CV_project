# CV_project
## Introduction
In this project we implement the main algorithm for plate detection which is formed by Yolov5 and pdlpr and then we compare it to a baseline method composed by traditional technique for plate area detection and CNN + CTC for plate character recognition

## Folder structure
- /dataset: here there are the images and the labels that are used for the training testing and validation, /cropped_images and /labels_pdlpr are used for the second part of character recognition in pdlpr and in the cnn + ctc method.
- /images: here there are the images generated during training by the models
- /models: the saved models that are then loaded for the testing phase
- /results: here there are txt files which contains the result metrics for the training and testing phase for each model

## File structure
