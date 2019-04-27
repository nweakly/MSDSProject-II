# Object Detection in the Home Video Security Systems

## Project overview
The goal of this project is to conduct a feasibility study of applying deep learning techniques for detecting and labeling objects in the video recordings of the home security systems, such as the Ring doorbell, in order to be able to eventually build a customizable home security system. There are many potential scenarios where a custom-tuned home security system could be useful, for example, to notify homeowners that their mail and packages have been delivered, to combat so-called “porch pirates”, to detect undesirable activity nearby (e.g., people displaying guns) or to let parents know that their children safely returned home after taking their family dog for a walk. 

The Ring doorbell records video clips when detecting motion within a predetermined perimeter. However, this motion sensor can be triggered not only by humans walking up to the door, but by wild and domestic animals, passing vehicles, etc. So, the first step of this project is using an algorithm capable of processing video feed in real (or near real) time to identify and classify objects, and then training the model to identify additional context-dependent objects in the video recordings (video feed).

For more detailed explanation of the project please watch: It contains some practical suggestions and lessons learned while working on this project.

## Technical Requirements and Dependencies:
- Anaconda package (64-bit version) on Windows 10
- Python 3.6
- TensorFlow (GPU version)
- OpenCV
- Cython extensions - Python to C compiler and wrapper to be able to call DarkNet C code from Python
- Jupyter Notebook
- DarkNet framework - original implementation if the YOLO algorithm written in C and CUDA by Joseph Redmon https://github.com/pjreddie/darknet
- Darkflow
- cfg (configuration) and weights files for the YOLO model downloaded from https://pjreddie.com/darknet/yolo/
- highly recommended to create a separate conda virtual environment (to resolve version conflicts for the deep learning libraries)
- GPU GeForce RTX 2070 used during model raining process, GeForce GTX1050 for all other file processing.

For detailed installation instructions please refere to a post by Abhijeet Kumar (https://appliedmachinelearning.blog/2018/05/27/running-yolo-v2-for-real-time-object-detection-on-videos-images-via-darkflow/ ) or 

## Project steps:
### Data collection

scrit to download acual Ring  (test videos)video clips
-scraping
taking, and annotating pictures
taking frames out of the actual video clips

### EDA and Data Preprocessing
resigins etc

### Fitting a pre-trained model
testing oon still images
testing on video

### Training a New Model on a Custom Data Set
Transfer learning approach:
1. chose a pre-traines model
2. change configurations to fit a particular situations
3. build the model
4. train the model
training is more effective in command-line mode
5. use the new model for predictions


### Use  the new model for predictions


### Interpretation of the Results

## Conclusions
- YOLOv2 model is not very accurate in predicting smaller objects
- quality of the training data maters a lot
- prediction of non-existing objects in previousely seen locations and non-detecting objects in unseen locations
- YOLO is ot very accurate when objects are partically hidden => more training data needs to reflect that

## References
Video presentation for this project: 
- Darklow library (Darknet translated to TensorFlow) https://github.com/thtrieu/darkflow
- Official site for the Darknet project https://pjreddie.com/darknet/yolo/ . Use to download configuration and pretrained wights files.
- Jay, Mark. Series of YOLO tutorials: https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM&index=1 and 
https://github.com/markjay4k/YOLO-series 
- Instructions for setting up YOLO using Anaconda and Windows https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html

Darknet framework https://github.com/pjreddie/darknet
