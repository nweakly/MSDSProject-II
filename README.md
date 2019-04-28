# Object Detection in the Home Video Security Systems

## Project overview

![Ring Doorbell](https://github.com/nweakly/MSDSProject-II/blob/master/Data/scene00106.png "Screen shot of the Ring Doorbell recording")

The goal of this project is to conduct a feasibility study of applying deep learning techniques for detecting and labeling objects in the video recordings of the home security systems, such as the Ring doorbell, in order to be able to eventually build a customizable home security system. There are many potential scenarios where a custom-tuned home security system could be useful, for example, to notify homeowners that their mail and packages have been delivered, to combat so-called “porch pirates”, to detect undesirable activity nearby (e.g., people displaying guns) or to let parents know that their children safely returned home after taking their family dog for a walk. 

The Ring doorbell records video clips when detecting motion within a predetermined perimeter. However, this motion sensor can be triggered not only by humans walking up to the door, but by wild and domestic animals, passing vehicles, etc. So, the first step of this project is using an algorithm capable of processing video feed in real (or near real) time to identify and classify objects, and then training the model to identify additional context-dependent objects in the video recordings (video feed).

For a more detailed explanation of the project please watch the video presentation. It contains a description of the project some practical suggestions and lessons learned while working on this project.

## Technical Requirements and Dependencies:
- Anaconda package (64-bit version) on Windows 10
- Python 3.5 or higher
- TensorFlow (GPU version preferred)
- OpenCV
- Cython extensions - Python to C compiler and wrapper to be able to call DarkNet C code from Python
- Jupyter Notebook
- DarkNet framework - original implementation of the YOLO algorithm written in C and CUDA by Joseph Redmon https://github.com/pjreddie/darknet
- Darkflow - package translating Darknet to TensorFlow
- cfg (configuration) and weights files for the YOLO model downloaded from https://pjreddie.com/darknet/yolo/
- highly recommended - a separate conda virtual environment (to resolve version conflicts for the deep learning libraries) and use Anaconda for installations
- GPU GeForce RTX 2070 used during model raining process, GeForce GTX1050 for all other file processing.

For detailed installation instructions please refere to a post by Abhijeet Kumar (https://appliedmachinelearning.blog/2018/05/27/running-yolo-v2-for-real-time-object-detection-on-videos-images-via-darkflow/ ) or https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html .

## Project steps:
### Data collection, EDA and Preprocessing
For this project, I assembled custom training and testing datasets using the following tools and data sources:
- video recordings (for testing and extracting still images) from a personal Ring device collected using DataCollection.ipynb ;
- static images (for training, testing, and presentation) extracted from the video recordings using VLC media player (for instructions see  https://www.raymond.cc/blog/extract-video-frames-to-images-using-vlc-media-player/);
- additional training images were scraped using Google image search using Training_data_collection.py script;
- video files preprocessed using DataPreprocessing.ipynb to decrease the size, discard audio and cut out necessary parts of the video recordings;
- additional training pictures of a crowbar were taken by the author of the project; 
- Annotating_images.py and Drawing_Boxes.py scripts were used to manually draw bounding boxes around crowbars (to train a custom model) and create xml files with image annotations; 
- additional data augmentation techniques were randomly applied to the training data set (rotation, flipping, scaling, translation, color saturation changes, and cropping) using Photoshop batch processing.  

The original video recordings from the Ring device have frame size 1920x1080 pixels with 15 frames per second rate.   In order to accommodate existing hardware better, the videos were downsized to 640x360 pixels while retaining 15.0 frames/second rate. 
The resulting set of training crowbar images collected from all sources and augmentation techniques applied includes 554 total images.

### Fitting a pre-trained model
Since the detection speed is a very important factor in processing security videos, among all available CNN approaches  I chose to use a one-stage detector model, namely the __YOLO ("You Only look Once") model__ originally introduced in 2016 in the paper written by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi.  The updated YOLOv2 algorithm was translated to Tensorflow by Trieu H. Trinh and is available as an open source __darkflow package__ (https://github.com/thtrieu/darkflow). 

A test example of YOLOv2 pretrrained model applied to a static image can be found at https://github.com/nweakly/MSDSProject-II/blob/master/YOLO_Model_Test.ipynb which successfully with 68.4% confidence identified a cat in the picture. 

Using YOLOv2 for predictions is easier accomplished through the command line interface, for example using the following command:

```
python flow --model cfg/yolov2.cfg --load bin/yolov2.weights --demo videofile.avi  --gpu 1.0 --saveVideo
```
 (notes:  before running this command, navigate to the darkflow master directory and download the cfg file and the weights file, for example from https://pjreddie.com/darknet/yolo/ maintained by the authors of the YOLO algorithm.  For more options refer to the official darkflow repository instructions https://github.com/thtrieu/darkflow )
 
Applied to the videos (please see mp4 files in Data/Processed folder),  the YOLOv2 and its smaller modification YOLOv2 tiny showed good detection results for large objects in both normal and low light conditions as long as there is an unobstructed view of an object. 

### Training a New Model on a Custom Data Set
Transfer learning approach:
1. chose a pre-traines model
2. change configurations to fit a particular situationsUsing YOLOv2 for predictions is easier accomplished through the command line interface, for example using the following command:


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
- Jay, M. Series of YOLO tutorials: https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM&index=1 and 
https://github.com/markjay4k/YOLO-series 
- Instructions for setting up YOLO using Anaconda and Windows https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html

Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). You Only Look Once: unified, real-time object detection. Retrieved from:  https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf 

Darknet framework https://github.com/pjreddie/darknet
