# Introduction
This repository contain all the work done while being an intern at UST Global during June-July 2020.

# Project
Object Detection frameworks - Understand available frameworks(YOLOv3) and build pipeline for customer object detection tasks.
In this project we needed to create a custom dataset(for this project only bags) and train it our yolov3 model on this dataset.

# How to create a custom dataset
There are many software available that can be used to create a custom dataset. But the one I have used in this project and found it very user frienly is labelImg.
A detailed tutorial on how to use labelImg can be found [here](https://www.arunponnusamy.com/preparing-custom-dataset-for-training-yolo-object-detector.html).

# Training
I have used Google Colab to train my model. You can find all the necessary files needed to run the model on colab in the [darknet](https://drive.google.com/file/d/1rww_EWsWSreUeB6ZU3lkA7lMUzN8IOIK/view?usp=sharing) folder saved on my drive. (The account of drive and colab should be same, so one would need to upload this folder on their drive before using it.
    For this project there was only one class (bags), if you are going to traing on more classes or some other class you would need to change the 'yolo.names', 'yolo.data' and  'yolov3_custom_train.cfg' files.
