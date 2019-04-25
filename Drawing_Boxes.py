# Drawing boxes for the target objects usinig cv2
# in order to create a training data set
# has to have "Annotating_images.py" saved in the same directory 

# Adaped from https://github.com/markjay4k/YOLO-series/blob/master/part6%20-%20draw_box.py
# by Mark Jay

import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from Annotating_images import write_xml

# global constants
img = None
tl_list = [] #top left coordinates
br_list = [] #bottom right coordinates
object_list = []

# constants
#specify the appropriate folder
image_folder = 'new_data'
savedir = 'annotations'
# specify the target class
obj = 'crowbar'


def line_select_callback(clk, rls):
    global tl_list
    global br_list
    global object_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj
	

def onkeypress(event):
    global object_list
    global tl_list
    global br_list
    global img
    if event.key == 'q':
        print(object_list)
        write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
        tl_list = []
        br_list = []
        object_list = []
        img = None
        plt.close()


def toggle_selector(event):
	toggle_selector.RS.set_active(True)


if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir('new_model/new_data')):
        img = image_file
        fig, ax = plt.subplots(1)
        image = cv2.imread(image_file.path)#read full path to the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert color
        ax.imshow(image) #display the image

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5, #button=[1] -left mouse click 
            spancoords='pixels', interactive=True
        )
        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event', onkeypress)
        plt.show()