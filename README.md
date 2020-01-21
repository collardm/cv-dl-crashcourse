17-day Computer Vision and Deep Learning Crash Course
==========================
> This crash course by Adrian Rosebrock from PyImageSearch is awesome for getting hands-on in computer vision and deep learning. It aims at teaching you the fundamentals with fast tutorials that solve real-world problems. 

* Link : https://www.pyimagesearch.com/welcome-crash-course/

# Table of Contents
0. [Installation](#ch0)
1. [Day 1 | Face Detection with OpenCV and deep Learning](#ch1)
1. [Day 2 | OpenCV Tutorial: A Guide to Learn OpenCV](#ch2)
1. [Day 3 | Mobile Document Scanner](#ch3)
1. [Day 4 | Multiple Choice Questions scanner using Optical Mark Recognition](#ch4)
1. [Day 5 | Object tracker](#ch5)
1. [Day 6 | Measuring the size of objetcs with OpenCV](#ch6)
1. [Day 8 | Facial landmarks with dlib and OpenCV](#ch8)
1. [Day 9 | Eye blink detection with OpenCV, Python and dlib](#ch9)
1. [Day 10 | Drowsiness detection](#ch10)
1. [Day 12 | Dogs vs. Cats : Feedforward neural networks with Keras](#ch12)
1. [Day 13 | Deep Learning with OpenCV](#ch13)
1. [Day 14 | Create custom DL datasets](#ch14)
1. [Day 15 | Train a Convolutional Neutal Networks with Keras](#ch15)

<a id="ch0"></a>
# Intallation
First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/collardm/cv-dl-crashcourse.git
    $ cd cv-dl-crashcourse

If you do not want to install git, you can instead download [master.zip](https://github.com/collardm/cv-dl-crashcourse/archive/master.zip), unzip it, rename the resulting directory to `cv-dl-crashcourse` and move it to your development directory.

If you are familiar with Python and you know how to install Python libraries, go ahead and install the libraries listed in `requirements.txt`. If you need detailed instructions, please read on.

## Python & Required Libraries
Of course, you obviously need Python. Python 3 is already preinstalled on many systems nowadays. You can check which version you have by typing the following command (you may need to replace `python3` with `python`):

    $ python3 --version  # for Python 3


Any Python 3 version should be fine, preferably 3.5 or above. If you don't have Python 3, I recommend installing it. To do so, you have several options: on Windows or MacOSX, you can just download it from [python.org](https://www.python.org/downloads/). On MacOSX, you can alternatively use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). If you are using Python 3.6 on MacOSX, you need to run the following command to install the `certifi` package of certificates because Python 3.6 on MacOSX has no certificates to validate SSL connections (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3 python3-pip


## Using pip
We need to install several Python libraries that are necessary for this project, in particular NumPy, Matplotlib, OpenCV, scikit-image and TensorFlow (and a few others). For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments. Moreover, the pip packages are usually the most recent ones available, while Anaconda and system packages often lag behind a bit.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries. Note: in all the following commands, if you chose to use Python 2 rather than Python 3, you must replace `pip3` with `pip`, and `python3` with `python`.

First you need to make sure you have the latest version of pip installed:

    $ python3 -m pip install --user --upgrade pip

The `--user` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use `sudo python3` instead of `python3` on Linux), and you should remove the `--user` option. The same is true of the command below that uses the `--user` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

    $ python3 -m pip install --user --upgrade virtualenv
    $ python3 -m virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace `` `which python3` `` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

    $ source ./env/bin/activate

On Windows, the command is slightly different:

    $ .\env\Scripts\activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using `sudo pip3` instead of `pip3` on Linux).

    $ python3 -m pip install --upgrade -r requirements.txt

* Note: You may want to remove dlib from the requirements.txt

Congrats! You are ready to learn Compute Vision and Deep Learning, hands on!

Next, jump to the first tutorial **Day 1 : Face detection with OpenCV and deep learning**

<a id="ch1"></a>
## Day 1 | Face Detection with OpenCV and deep learning

Link to the blog post: <https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning>


In your virtual environment you can run this command to perform face detection on an image of your choice :

    $ python 1-face-detection-dl/detect_faces.py --image images/rooster.jpg \
    --prototxt model/deploy.prototxt.txt \
    --model model/res10_300x300_ssd_iter_140000.caffemodel
    
And for using the Webcam :

    $ python 1-face-detection-dl/detect_faces_video.py --prototxt model/deploy.prototxt.txt \
    --model model/res10_300x300_ssd_iter_140000.caffemodel

<a id="ch2"></a>
## Day 2 | OpenCV Tutorial: A Guide to Learn OpenCV

Link to the blog post: <https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv>

Fundamentals of the basics image processing using OpenCV, the world's most popular computer vision library :

    $ python 2-opencv-tutorial/opencv_tutorial_01.py

Counting objects in an image uses a lots of processing technique like edge detection, masking and bitwise operations, etc.. :

    $ python 2-opencv-tutorial/opencv_tutorial_02.py --image images/tetris_blocks.png

<a id="ch3"></a>
## Day 3 | Mobile Document Scanner

Link to the blog post: <https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes>

A practical application of edge detection to find the Region Of Interest and perspective transform to correctly view the document :

    $ python 3-document-scanner/scan.py --image 3-document-scanner/images/page.jpg

<a id="ch4"></a>
## Day 4 | Multiple Choice Questions scanner using Optical Mark Recognition

Link to the blog post: <https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv>

This is a bubble sheet scanner and test grader.

    $ python 4-omr-on-mcq/test_grader.py --image 4-omr-on-mcq/images/test_01.png

<a id="ch5"></a>
## Day 5 | Object tracker

Link to the blog post: <https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv>

This code detect the presence of a colored ball using computer vision techniques and then track the ball as it moves.

    $ python 5-ball-tracking/ball_tracking.py --video 5-ball-tracking/ball_tracking_example.mp4

<a id="ch6"></a>
## Day 6 | Measuring the size of objetcs with OpenCV

Link to the blog post: <https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv>

Thanks to the width of the left-most object in the image (in inches), our reference object, we can dertermine the size of any object in an image.

    $  python 6-measuring-size-objetcs/object_size.py --image 6-measuring-size-objetcs/images/example_01.png \
    --width 0.955 

<a id="ch8"></a>
# Day 8 | Facial landmarks with dlib and OpenCV

Link to the blog post : <https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python>

For this tutorial, you will need to install and configure dlib. I recommend this post [How to install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) by Adrian Rosebrock.
For Windows : 
* Install Visual Studio 2019
* Install CMake
* Run install script below 

<details>
    <summary>Click me to show code.</summary>
    <pre>
        <code>
$ git clone https://github.com/davisking/dlib.git
$ cd dlib-19.17
$ git checkout tags/v19.17mkdir build

$ mkdir build
$ cd build

$ cd buildcmake -G "Visual Studio 16 2019" -A x64 \
-DJPEG_INCLUDE_DIR=..\dlib\external\libjpeg \
-DJPEG_LIBRARY=..\dlib\external\libjpeg \
-DPNG_PNG_INCLUDE_DIR=..\dlib\external\libpng \
-DPNG_LIBRARY_RELEASE=..\dlib\external\libpng \
-DZLIB_INCLUDE_DIR=..\dlib\external\zlib \
-DZLIB_LIBRARY_RELEASE=..\dlib\external\zlib \
-DCMAKE_INSTALL_PREFIX=install ..

$ cmake --build . --config Debug --target INSTALL
</code>
    </pre>
</details>

Finally, go to the folder dlib-19.17 (for me) and execute :

    $ python setup.py install

To run the code :

    $ python 8-facial-landmarks/facial_landmarks.py \
    --shape-predictor 8-facial-landmarks/model/shape_predictor_68_face_landmarks.dat \
    --image 8-facial-landmarks/images/example_03.jpg

<a id="ch9"></a>
## Day 9 | Eye blink detection with OpenCV, Python and dlib

Link to the blog post : <https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib>

Eye blinking is a tricky combination of severals image processing methods.

To access to your webcam :

    $ python 9-blink-detection/detect_blinks.py \
    --shape-predictor 9-blink-detection/model/shape_predictor_68_face_landmarks.dat

For the example video execute the following command (make sure to uncomment the correct line, ad detailed in the code):

    $ python 9-blink-detection/detect_blinks.py \
	--shape-predictor 9-blink-detection/model/shape_predictor_68_face_landmarks.dat \
	--video 9-blink-detection/videos/blink_detection_demo.mp4


<a id="ch10"></a>
## Day 10 | Drowsiness detection

Link to the blog post: <https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv>

Now we know how detect eye blink we can extend it to a drowsiness detection.

    $ python 10-detect-drowsiness/detect_drowsiness.py \
    --shape-predictor 10-detect-drowsiness/model/shape_predictor_68_face_landmarks.dat \
    --alarm 10-detect-drowsiness/sounds/alarm.wav

<a id="ch12"></a>
## Day 12 | Dogs vs. Cats : Feedforward neural networks with Keras

Link to the blog post: <https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras>

To train the neural network (download the train dataset from the kaggle competition Dogs vs. Cats) :

    $ python 12-simple-neural-network/simple_neural_network.py \
    --dataset 12-simple-neural-network/kaggle_dogs_vs_cats \
    --model 12-simple-neural-network/output/simple_neural_network.hdf5

To test our neural network :

    $ python 12-simple-neural-network/test_network.py \
    --model 12-simple-neural-network/output/simple_neural_network.hdf5 \
	--test-images 12-simple-neural-network/test_images

<a id="ch13"></a>
## Day 13 | Deep Learning with OpenCV

Link to the blog post: <https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv>

OpenCV provides pre-trained convolutional neural network on ImageNet dataset.

    $ python 13-dl-cv2/dl_opencv.py --image 13-dl-cv2/images/jemma.png \
    --prototxt 13-dl-cv2/model/bvlc_googlenet.prototxt \
    --model 13-dl-cv2/model/bvlc_googlenet.caffemodel \
    --labels 13-dl-cv2/model/synset_words.txt

<a id="ch14"></a>
## Day 14 | Create custom DL datasets

Link to the blog post: <https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset>

Deep Learning is all about DATA, so let's see how to create a cumtom dataset.

<a id="ch15"></a>
## Day 15 | Train a Convolutional Neutal Networks with Keras

Link to the blog post: <https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns>

To train : 

    $ python 15-cnn-keras/train.py --dataset 15-cnn-keras/dataset \
    --model 15-cnn-keras/pokedex.model --labelbin 15-cnn-keras/lb.pickle

To test:

    $ python 15-cnn-keras/classify.py --model 15-cnn-keras/pokedex.model \
    --labelbin 15-cnn-keras/lb.pickle --image 15-cnn-keras/examples/charmander_counter.png
