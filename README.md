# Object Detection in the Home Video Security Systems

## Project overview
The goal of this project is to conduct a feasibility study of applying deep learning techniques for detecting and labeling objects in the video recordings of the home security systems, such as the Ring doorbell, in order to be able to eventually build a customizable home security system. There are many potential scenarios where a custom-tuned home security system could be useful, for example, to notify homeowners that their mail and packages have been delivered, to combat so-called “porch pirates”, to detect undesirable activity nearby (e.g., people displaying guns) or to let parents know that their children safely returned home after taking their family dog for a walk. 
The Ring doorbell records video clips when detecting motion within a predetermined perimeter. However, this motion sensor can be triggered not only by humans walking up to the door but by wild and domestic animals, passing vehicles, etc. So, the first step of this project is using an algorithm capable of processing video feed in real (or near real) time to identify and classify objects, and then training the model to identify additional context-dependent objects in the video recordings (video feed).
For more detailed explanation of the project please watch:




## References
Video presentation for this project: 
- Darklow library (Darknet translated to TensorFlow) https://github.com/thtrieu/darkflow
- Official site for the Darknet project https://pjreddie.com/darknet/yolo/ . Use to download configuration and pretrained wights files.
- Jay, Mark. Series of YOLO tutorials: https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM&index=1 and 
https://github.com/markjay4k/YOLO-series 
- Instructions for setting up YOLO using Anaconda and Windows https://expschoolwork.blogspot.com/2018/11/using-yolo-in-anaconda.html

