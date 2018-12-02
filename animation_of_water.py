import numpy as np
import keras
import tensorflowjs as tfjs
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
import aifc
import scipy.signal as sp
import wave
import scipy
import pylab as pl
import os
from PIL import Image
import math
from skimage.transform import resize
from sklearn.feature_extraction import image
from sklearn.preprocessing import MinMaxScaler, normalize
from matplotlib import mlab
from keras import layers
from keras.layers import TimeDistributed, Permute, Conv3D, ConvLSTM2D, Conv3DTranspose, Conv2DTranspose, UpSampling2D, AveragePooling2D, MaxoutDense, Bidirectional, Input, RepeatVector, Dot, Embedding, Lambda, LSTM, GRU, Dense, Reshape, Dropout, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Concatenate, MaxPooling3D, LeakyReLU
from keras.models import load_model, Sequential, Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import plot_model
from keras.engine.topology import Layer, InputSpec
from keras.utils import np_utils
from matplotlib.pyplot import imshow
import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_last')

def extractFrames(inGif, maxlen, size):
    ''' extractFrames Method
            Extract image frames from a GIF
            
            Args: 
                inGif: string file of GIF to read 
                maxlen: int maximum number of frames per GIF
                size: tuple of dimensions for each image frame
            Returns:
                4-D volume of image frames if file can be read, else 0    
                
    '''
    try:
        # Open GIF using Image module from PIL
        frame = Image.open(inGif)
        # Counter for number of frames read
        nframes = 0
        # Initialize 4-D volume of image frames, where the first dimension corresponds to
        # the number of frames, the second dimension corresponds to the number of rows in
        # an image frame, the third dimension corresponds to the number of columns in an
        # image frame, and the fourth dimension is the channels axis
        frame_volume = np.zeros((maxlen,size[0],size[1],1),dtype=np.uint8)
        while frame:
            # Upon entering the while loop, the pointer is set to the first image frame
            # Resize image frame to the desired input size using an antialias filter
            single_frame = frame.resize(size,Image.ANTIALIAS)
            # Remove transparency information if present
            if 'transparency' in single_frame.info.keys():
                del(single_frame.info['transparency'])
            # Convert image frame to a numpy array
            single_frame = np.array(single_frame)[...,np.newaxis]
            # Add image frame to frame_volume
            frame_volume[nframes,:,:,:] = single_frame
            # If maxlen image frames are read break out of the while loop, else move the
            # pointer to one image frame forward
            nframes += 1
            if nframes == (maxlen):
                break
            try:
                frame.seek( nframes )
            # If the incremented pointer is not pointing at a valid image frame (i.e. if
            # the pointer has moved past the last valid image frame), then execute the 
            # following:
            except EOFError:
                # If there is only one frame in the GIF, the GIF is not valid. Ensure the
                # function returns 0 instead of the 4-D frame volume
                if nframes == 1:
                    frame_volume = 0
                # If the number of valid frames in the GIF is less than the number of desired
                # frames, incrementally move backward (to the beginning of the set of GIF frames)
                # and forward again if necessary (to the end of the set of GIF frames) until the 
                # total number of desired frames is reached. In this way, although frames from
                # the original set of GIF frames are repeated, every frame volume is guaranteed to
                # contains the same number (maxlen) of frames
                elif nframes <= (maxlen-1) and nframes != 1:
                    right_pointer = nframes
                    left_pointer = nframes-1
                    flag = 0
                    while right_pointer < (maxlen):
                        frame_volume[right_pointer,:,:,:] = frame_volume[left_pointer,:,:,:]
                        if left_pointer == 0:
                            flag = 1
                        elif left_pointer == nframes:
                            flag = 0
                        if flag == 0:
                            left_pointer -= 1
                        elif flag == 1:
                            left_pointer += 1
                        right_pointer += 1
                break;
        return frame_volume
    # If the GIF cannot be read return 0 instead of the frame volume
    except (OSError, TypeError, ValueError) as error:
        return 0

def masking(X_train):
    ''' masking Method
            Convert image frames to binary masks with (1) indicating a pixel has changed magnitude
            and (0) indicating a pixel has retained the same magnitude. 
            
            Args: 
                X_train: 5-D dataset of image frames
            Returns:
                5-D dataset of binary masks
                
    '''
    # Instantiate an array of zeros the same size as X_train 
    X_train_n = np.zeros_like(X_train[:,0:X_train.shape[1]-1,:,:,:])
    # A binary mask of an image frame is created by comparing (elementwise) the current image frame 
    # with the next image frame in sequence. If the value of a pixel remains the same, it is replaced
    # by a 0. If the value of a pixel changes, it is replaced by a 1.
    for ii in range(X_train.shape[0]):
        for jj in range(X_train.shape[1]-1):
            temp = (X_train[ii,jj,:,:,:] != X_train[ii,jj+1,:,:,:])*1
            X_train_n[ii,jj,:,:,:] = temp
    return X_train_n 

def pre_processing(X_train_n):
    ''' pre_processing Method
            Pre-process 5-D dataset of binary masks into a training set and set of ground truth
            annotations
            
            Args: 
                X_train_n: 5-D dataset of binary masks
            Returns:
                X_train_f: 5-D training set of binary masks
                Y_train_f: 5-D training set of ground truth binary masks
                
    '''
    # In the original 5-D training set, the first axis corresponded to GIFs and the second axis 
    # corresponded to frames from each GIF. Compress the data in these two axes into one axis in the 
    # output 5-D matrix. In this way, the first axis in the 5-D output matrix corresponds to binary 
    # masks from all GIFs.
    # Instantiate 5-D arrays of zeros of the desired shapes. X_train_f is the input to the neural
    # network.
    X_train_f = np.zeros((X_train_n.shape[0]*X_train_n.shape[1]-1,1,X_train_n.shape[2],X_train_n.shape[3],X_train_n.shape[4]))
    # The neural network will be tasked with taking as input a binary mask of an image frame and
    # predicting the binary mask of the next image frame in sequence. Y_train_f are the ground
    # truth binary masks for those predictions. (While X_train_f is composed of the first frame
    # from each GIF to the second-to-last frame, Y_train_f is composed of the second frame from
    # each GIF to the last frame. As a shifted version of X_train_f one time step forward,
    # Y_train_f therefore contains ground truth binary masks for predictions made on X_train_f). 
    Y_train_f = np.zeros_like(X_train_f)
    counter = 0
    for ii in range(X_train_n.shape[0]):
        for jj in range(X_train_n.shape[1]-1):
            X_train_f[counter,0,:,:,:] = X_train_n[ii,jj,:,:,:]
            Y_train_f[counter,0,:,:,:] = X_train_n[ii,jj+1,:,:,:]
            counter += 1
    return X_train_f,Y_train_f

# Neural Network model is based on Deep Convolutional Generative Adversarial Networks (DCGANs).
# In DCGANs, a Generator learns to create fake images and a Discriminator learns to differentiate
# between the fake images of the Generator and real images of the dataset. During training, the
# two models will compete with each other to become better at their respective tasks, the 
# Generator in particular using feedback (loss) from the Discriminator to learn how to synthesize
# more realistic images. After training is complete, the trained Generator can then be used
# to create new images from the state space.
# The Discriminator in DCGANs is typically implemented as a deep convolutional neural network, 
# whereby input images are passed through several convolutional layers and downsampled via strided 
# convolutions until the output layer, where a scalar probability value (that the image is real 
# vs fake) is given. 
# The Generator is typically implemented by taking as input a vector of noise and upsampling 
# the vector via transposed convolutions until an image of the appropriate dimensions is constructed
# by the output layer.
# In Conditional GANs (CGANs), by contrast, the Generator learns to generate a fake image based on
# a specified condition (such as class label, text, etc.) instead of a fake image based on a noise 
# distribution. In the case where the condition is an image, the Generator can first pass the image
# through several convolutional layers to extract a compressed, latent representation. Upsampling
# via transposed convolutions can then be used to create the new fake image (conditioned on the
# information extracted from the input "condition" image).
# Since the problem at hand only requires the neural network to predict the next binary mask from
# a previous binary mask, a simplified version of this procedure is replicated in the neural network
# below.
# In the neural network, tbe previous binary mask is given as the "condition" image and passed through
# three convolutional layers to yield a downsampled, latent representation. Transposed convolutions
# are then used to upsample the latent representation. The output layer then gives a prediction for
# the next binary mask in sequence. (Since the output layer uses a sigmoid activation function, the
# outputted image consists of probabilities, where the value for each pixel represents the probability
# that the pixel has changed value). The loss can be evaluated between the probability prediction
# and ground truth binary mask. 

def create_model():
    ''' create_model Method
            Create Neural Network model
            
            Returns:
                model: Compiled neural network
                
    '''
    dropout = 0.4
    input_shape=(64, 64, 1)
    X_input = Input(shape=input_shape)
    X = Conv2D(10,7,strides=2,padding='same')(X_input)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(dropout)(X)
    X = Conv2D(20,7,strides=2,padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(dropout)(X)
    X = Conv2D(40,7,strides=2,padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = Dropout(dropout)(X)
    X = UpSampling2D()(X)
    X = Conv2DTranspose(20,7,padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = UpSampling2D()(X)
    X = Conv2DTranspose(10,7,padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = UpSampling2D()(X)
    X = Conv2DTranspose(1,7,padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('sigmoid')(X)
    model = Model(X_input,X)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# Maximum number of frames to extract from each GIF
maxlen = 9
# Desired size of each image frame
size = (64,64)
# The first axis of the dataset X_train indicates the GIF to which the frames belong
# volume_index is an index into the first axis of the dataset
volume_index = 0
# Path to GIF files
path = 'Documents/GIFs/waterfall'
filenames = glob.glob(path+'/*.gif')
# Instantiate 5-D rray of zeros of the desired shape for the dataset set X_train
X_train = np.zeros((len(filenames),maxlen,size[0],size[1],1),dtype=np.uint8)
# Extract image frames from GIFs using extractFrames(). 
for ii in range(len(filenames)):
    frame_volume = extractFrames(filenames[ii], maxlen, size)
    # Error checking to see whether extractFrames() outputs a valid frame volume or 0
    if hasattr(frame_volume, "__len__"):
        X_train[volume_index,:,:,:,:] = frame_volume
        volume_index += 1
    else:
        X_train = X_train[:-1,...]

# Convert the dataset set of image frames to a dataset of binary masks
X_train_n = masking(X_train)

# Pre-process the dataset of binary masks into a training set and set of ground truth annotations
X_train_f,Y_train_f = pre_processing(X_train_n)
# Reshape the 5-D training set (X_train_f) and 5-D set of ground truth annotations (Y_train_f) into
# 4-D matrices for training
X_train_f = X_train_f.reshape((X_train_f.shape[0],X_train_f.shape[2],X_train_f.shape[3],X_train_f.shape[4]))
Y_train_f = Y_train_f.reshape((Y_train_f.shape[0],Y_train_f.shape[2],Y_train_f.shape[3],Y_train_f.shape[4]))

# Create neural network model and train
model = create_model()
model.fit(X_train_f,Y_train_f,batch_size=64,epochs=100)

# Save the model architecture and weights
model.save('tfjsmodel.h5')
# Convert the model architecture and weights to a form that can be loaded into Tensorflow JS
tfjs.converters.save_keras_model(model, "tfjsmodel")