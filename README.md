Animation of Water
=========================

AI learns how to animate water

Deep learning has been famously used for image classification and sequence generation, making it an exceptional tool for learning scene dynamics in videos. A variety of methodologies have been applied to this task in recent years, including recurrent neural networks, as in [Recurrent Neural Networks for Modeling Motion Capture Data](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/wpapers/DL2.pdf) and generative adversarial networks, as in [Generating Videos with Scene Dynamics](http://www.cs.columbia.edu/~vondrick/tinyvideo/). The approach taken in this application is a synthesis of both works:

- In [Recurrent Neural Networks for Modeling Motion Capture Data](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/wpapers/DL2.pdf), the authors use hours of motion capture data, essentially time series of changes in 3D position coordinates for 64 different joints. The training set was constructed by partitioning the time series into pairs of frames X and Y, where Y is the next frame in the sequence of motion that follows X. In this way, an RNN trained on this dataset learns to complete a sequence of motion given one frame at a time. 

- In [Generating Videos with Scene Dynamics](http://www.cs.columbia.edu/~vondrick/tinyvideo/), a generative adversarial network (GAN) was trained on large quantities of short videos to learn a realistic model of scene dynamics. The GAN is optimized through a min-max game between a generator model (that attempts to produce a realistic video) and a discriminator model (that attemps to differentiate between the fake videos of the generator and real videos of the dataset). 

In this application, the dataset consists of a large quantity of GIFs of waterfalls downloaded from Google Images. The GIFs were converted into individual frames and these frames subsequently converted into binary masks, where a value of (1) indicates a pixel has changed magnitude and a value of (0) indicates a pixel has retained the same magnitude. In this way, the training set is composed of pairs of binary masks X and Y, where Y is the next binary mask in the sequence of motion that follows X. A neural network trained on this dataset therefore learns the dynamics of water - given an input image frame, it will be able to predict which pixels in the image frame change and which do not. (This simplified framework for the dataset is justified, in the sense that only a limited group of pixels in a GIF will change value to indicate movement in the scene, with the rest of pixels remaining the same to display the underlying background. For example, in a GIF of a waterfall, pixels corresponding to the waterfall in the image and pixels adjacent to the waterfall will change to show the water flowing downward, while pixels for the trees and cliffside will not change value). 

The trained neural network will be able to predict the movement of water on a pixel-by-pixel level using these binary masks. A black canvas is drawn to screen and the initial starting location of water and its spatial extent are determined randomly, with water indicated by blue pixels on the black canvas. 

<img src="https://github.com/cchinchristopherj/Animation-of-Water/blob/cchinchristopherj-patch-1/Images/animation1.png" width="400" height="450" />

A binary mask of this initial scene (with 1 indicating a blue pixel of water and 0 indicating the black canvas background) will be given as input to the neural network, which will predict the next binary mask. Values of 1 in this binary mask will be changed into blue pixels and values of 0 will be changed into black pixels. 

<img src="https://github.com/cchinchristopherj/Animation-of-Water/blob/cchinchristopherj-patch-1/Images/animation2.png" width="400" height="450" />

This process will repeat to generate the water scene dynamics and the "Reset" button can be pressed to start a new scene.

<img src="https://github.com/cchinchristopherj/Animation-of-Water/blob/cchinchristopherj-patch-1/Images/animationfinal.png" width="400" height="450" />    

If you wish to replicate these results and train your own neural network, follow the subequent sets of instructions: 

Modules and Installation Instructions
=========================

**"Standard" Modules Used (Included in Anaconda Distribution):** numpy, matplotlib, pylab, glob, aifc, os, scipy, PIL, math

If necessary, these modules can also be installed via PyPI. For example, for the "numpy" module: 

        pip install numpy

**Additional Modules used:** skimage, sklearn, keras, tensorflowjs, google_images_download

skimage, sklearn, and tensorflowjs can be installed via PyPI. For example, for the "scikit-image" module:

        pip install scikit-image

For Keras, follow the instructions given in the [documentation](https://keras.io/#installation). Specifically, install the TensorFlow backend and, of the optional dependencies, install HDF5 and h5py if you would like to load and save your Keras models. 

For google_images_download, follow the instructions given in the [documentation](https://github.com/hardikvasa/google-images-download). While a normal installation of google_images_download simply requires PyPI, in order to obtain the large quantities of images necessary to train a neural network, the Selenium library and chromedriver must also be installed. See the troubleshooting section for more information. 

Correct Usage
=========================

In order to create the dataset, the google_images_download module uses the command line interface to download google images corresponding to keyword searches.

From Terminal (on Mac) or Command Utility (on Windows), run the following command:

   googleimagesdownload --keywords "waterfall" --format gif --related_images

A Google search is performed for "waterfall" images of type "gif" and all obtained images are downloaded to the local hard disk. (Additional arguments can be given to specify the location on the hard disk you wish to save the files and/or apply additional constraints on the type of images obtained). For instance, the "related_images" argument allows images related to the provided keyword to be searched for as well, thereby significantly increasing the total number of images obtained.

In order to replicate the process used to train the neural network, identify the directory on your hard disk where the newly-downloaded dataset of images is located. Change the "path" variable in line 238 of animation_of_water.py to this directory. Then run:

    python animation_of_water.py
    
This trains the weights of the neural network on the dataset and saves the architecture and weights of the model to your hard disk as both a Keras HDF5 file and tensorflow.js JSON and binary weight shard file(s). 
