# SignLanguageInterpreter

Language Used: Python

# ABSTRACT
Our goal is to create an application that takes input from the user in the form of a photo or video, interprets and translates the sign language symbols represented in the given input and outputs the given translation in the form of text or speech. We analyze existing solutions for the communication barrier between HIIs and other people such as text to speech algorithms and aim to improve on them by providing better ease of use and accuracy.


# MODULES USED
MediaPipe Holistic <br/>
MediaPipe Holistic is an open-source machine learning pipeline from Google that provides a set of pre-built models and tools for performing human pose estimation, hand tracking, and face detection and landmark estimation. It is designed to process video streams in real-time and can be used for a variety of applications, such as augmented reality, virtual try-on, and fitness tracking.
<br/><br/>Matplotlib module <br/>
Matplotlib is a data visualization library for creating static, animated, and interactive visualizations in Python. It provides a variety of plotting functions for creating line plots, scatter plots, bar plots, histograms, and more. Matplotlib can be used to generate publication-quality figures and can be customized in a variety of ways.
<br/><br/>OpenCV Model <br/>
OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision tasks. It includes a variety of image and video processing algorithms, such as feature detection, object recognition, and motion tracking. OpenCV is widely used in applications such as robotics, surveillance, and augmented reality.
<br/><br/>NumPy Module <br/>
NumPy is a numerical computing library for Python that provides support for arrays and matrices. It includes a variety of mathematical functions for manipulating arrays, such as element-wise operations, matrix multiplication, and linear algebra operations. NumPy is often used in scientific computing, data analysis, and machine learning applications.





# HOW IT WORKS
Step 1. Extract holistic key points <br/>
We build up this project step by step to be able to detect a bunch of different poses and specifically sign language signs. In order to do that we're going to be using a few key models. The primary model for our project is MediaPipe Holistic. Its function is to extract key points so this is going to allow us to extract key points from our hands, from our body and from our face, since that is mainly used in sign-language detection. 
We start with collecting a bunch of data on all of our different key points, and saving each of those collected data points as numpy arrays (using the NumPy module).

<br/><br/>Step 2. Train an LSTM Deep Learning Model <br/>
Using Tensorflow and Keras, build up a LSTM model to be able to predict the action which is being shown on the screen. In this particular case, the actions are going to be sign language signs, so we're going to use that LSTM model to do that. 
To be specific, we are basically training a Deep Neural Network (DNN) with LSTM Layers for sequences: train a DNN using LSTM Layers to go ahead and predict temporal components. We then would be able to predict the action from a number of frames, and not just a single frame, as previously stated.

<br/><br/>Step 3: Make real time predictions using Sequences. <br/>
Take MediaPipe Holistic, including our trained LSTM model and actually go on ahead and predict signs in real time. 
How it works: We utilize the OpenCV module for this, connecting all the components together with the help of this module, predicting in real time using a webcam. 

















# ALGORITHM/ STEPS OF IMPLEMENTATION
Import and Install Dependencies <br/>
We install the necessary Tensorflow and Keras Modules necessary for the execution of our project, and import all the stated modules given above (i.e. MatplotLib, NumPy, MediaPipe Holistic and OpenCV). 
<br/><br/>Keypoints using MP Holistic <br/>
We initialize the Mediapipe functions in order to extract keypoints of our face, left and right hands and pose, and initialize functions that are used to draw landmarks on the displayed frames based on these keypoints
<br/><br/>Extract Keypoint Values <br/>
We extract the key points from our body, for instance, all of the joints within our hands, and all of the joints within our body and our face. We then export these very same keypoints, which will represent our different frames at a point in time for our LSTM model. 
<br/><br/>Setup Folders for Collection <br/>
We set up local folders and directories where the collected data is to be stored.
<br/><br/>Collect Keypoint Values for Training and Testing. <br/>
We collect the data required in order to train the model using OpenCV to read the incoming input of the user’s camera, and Mediapipe and the previously defined functions in order to extract the keypoints corresponding to each sign language action and store them in the folders set up before
<br/><br/>Preprocess Data and Create Labels and Features <br/>
We read the data from the folders where they are stored and split the given dataset into a training dataset in order to train our model, and a testing dataset in order to verify its accuracy.
<br/><br/>Build and Train LSTM Neural Network <br/>
We implement a Sequential Tensorflow model and add LSTM and Dense layers to it, and then proceed to train this constructed model on the train dataset.
<br/><br/>Save Weights <br/>
After training the dataset, we save the weights of the given dataset locally.
<br/><br/>Evaluation using Confusion Matrix and Accuracy <br/>
We make use of a multilabel confusion matrix and accuracy score in order to test the accuracy of the given model. We use the model to predict the values of the test dataset and then compare them with the actual values using the above mentioned metrics.
<br/><br/>Test in Real Time <br/>
Finally, we apply the model in real time onto a live video capture feed of the user using OpenCV. We extract each frame of the user’s webcam and pass it to the model in real time and use this data to detect what sign the user is performing.

