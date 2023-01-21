# ECE491, Special Problems

## Introduction
The main goal of this research is to recover signals in the presence of Speckle Noise. The way this can be done is using Machine Learning, Autoencoders, and algorithms based off that. My approach, code, algorithms, and process will be outlined below.

### Motivation 
There are a wide range of computational imaging systems that illuminate the object of interest
by coherent light. Examples of such imaging systems are synthetic aperture radar (SAR) and
digital holography. These systems all suffer from a very particular type of noise referred to as
speckle noise. Speckle noise is very different and than typical additive noise, since it is
multiplicative. Most solutions to tackle speckle noise seem to be heuristic. The main motivation
for this work is the recent paper by Prof. Jalali and her collaborators that looks into such
systems from a theoretical perspective and shows that compressed sensing in the presence of
speckle noise is possible. However, the results of that paper are mainly theoretical. In this
project, we are interested in taking advantage of powerful tools from machine learning, such as
auto encoders, to implement algorithms that recover a signal in the presence of speckle noise.

### Concepts / Tools being used
The problem of signal recovery from noisy samples that are corrupted by speckle noise is
inherently very complex mathematically. To be successful in this project, I need to understand
those concepts and then, based on the results of that paper and other prior work on application
of compressed sensing, implement a proper recovery algorithm.

Most of my coding will be done in Python3 and the main libraries that I will be using are Numpy,
Numba, and PyTorch. Numba is going to be used for a wide variety of operations such as
speeding up calculations and will translate the code to be easily-compilable. Numpy is the basic
mathematical calculation library that I will be using to work with doing operations on vectors.
PyTorch is the biggest library that I will use since it is the main machine learning tool that I have.
I’ll be using that to train and use neural networks that I have to make. Also, PyTorch is an
important skill to learn as an ECE undergraduate because of its wide use and doing this project
would help me.

## Thesis 
The use of advanced signal processing techniques, such as denoising combined with machine learning algorithms can significantly improve the accuracy of signal recovery from noisy image samples of lungs with pneumonia, leading to more accurate diagnosis and treatment of the disease.

## The problem

Speckle noise is a type of noise that is commonly found in images acquired from optical imaging systems, such as laser imaging, ultrasound imaging and synthetic aperture radar (SAR) imaging. It is characterized by a granular or "salt-and-pepper" appearance, with small bright or dark spots scattered throughout the image.

<p align="center">
  <img 
    width="544"
    height="308"
    src="https://user-images.githubusercontent.com/101938119/213531959-61496e6f-abce-4d1b-a72d-9b925ac867f8.png"
  >
</p>

> Our goal here is to get from an image that looks like the one on the right.

It is caused by the constructive and destructive interference of light waves scattered by small, randomly distributed scatterers within the imaging system. The scattered waves combine to create a speckled pattern on the image. There can be different variations of noise disturbances such as multiplicative noise and additive nosie.

Speckle noise can be a significant problem in certain imaging applications, as it can reduce image quality, make it difficult to detect small features, and limit the ability to extract useful information from the image. Getting rid of it can be difficult but can prove to be very useful. As the thesis mentions, it can "[lead] to more accurate diagnosis and treatment of the disease".


## Graphical Depiction

In this case, we are talking about images and images having speckle noise, but I would first like to show speckle noise's effect on signals. Since images can be represented as vectors, this would be a good way to visualize what is happening. Since I am going to be converting the image we are working with to patches and then later to a vector, it is good to see what changes happen to a single vector. 

The noise formula is modeled by the following: $$\textbf{y} = AX_o\textbf{w} + \textbf{z}$$

Here is what all the variables are: 
- $\textbf{y}$ : This is the final measurement of the signal we end up with, and is the one that we see. 
- $A$ : This is a multiplicative constant (can be in the form of a matrix).
- $X_o$ : This is the original signal in the form of a matrix. The signal elements are on the diagonal of a square matrix.
- $\textbf{w}$ : The speckle noise, this is the main issue we are dealing with. 
- $\textbf{z}$ : This is the white gaussian additive noise. 

### Test vector
We can generate random values for speckle noise, additive white gaussian noise, and the original signal to see a sample comparison between $X_o$ and $y$. 

We can take $w$ as some random multiplicative values, ranging between 0.8-1.2 because we don't want too much difference, but just enough to see. 
We can take $X_o$ as the matrix of the array [5.4, 7.65, 9.4, 3.4] as spread along its diagonal. This was randomly generated.
We can take $z$ as a guassian random distributed noise:
Ignoring $A$ for now, we can just set it equal to 1.

Here, I am converting all of the lists to python vectors so that we can do operations on them.

The way to do this in python is the following: 

- Let's assume that the randomly generated vector is a column vector (5x1). 

```
import numpy as np

# Generate a 5x1 vector (this can be our Xo - original vector) 
x = np.random.rand(5, 1)
print("Original Vector:")
print(x)

# Generate multiplicative constant in the form of a matrix (must be 5x5 because of our conditions) with random values between 0.1 and 1.9
A_matrix = np.random.uniform(low=0.1, high=1.9, size=(5, 5))
x = x * A_matrix
print("Vector with multiplicative constant included")
print(x)


# Generate multiplicative noise (this is the speckle part)
multiplicative_noise = 0.5
x = x * (1 + multiplicative_noise * np.random.randn(5, 1))
print("Vector with multiplicative noise:")
print(x)

# Generate additive noise
additive_noise = 0.1
x = x + additive_noise * np.random.randn(5, 1)
print("Vector with additive noise:")
print(x)
```

## Test Images
As aforementioned, I used 4 very high quality images of lungs with pneumonia. Here is one of them - they all look pretty similar from a third perspective. 

<p align="center">
  <img 
    width="544"
    height="308"
    src="https://user-images.githubusercontent.com/101938119/213531959-61496e6f-abce-4d1b-a72d-9b925ac867f8.png"
  >
</p>

### Image Patches

Before moving further in the program, we would need to convert the testing images that I have into smaller patches. There are two reasons that we would have to do this. 

- **Computational Power.** The main reason to change to patches is because the machines we are using cannot handle/support that much computation. Even after running our code with many patches from a singular high quality image, it took my machine (lenovo with i5) about 17.5 minutes to run the autoencoder model. Having to run multiple of models for multiple high quality images would be too much. GPU would be needed for this case. 

- **Efficiency.** Making patches like this is decreasing computational power for a similar accuracy. We would want to do this if we are working without funding / low budget. 

### Test patches
Given an image like the one above, we can get *n* number of patches from that specific image using the *sklearn* library. 

```
from sklearn.feature_extraction import image

# Load the image
image = ...

# Define the patch size (e.g. (64, 64))
patch_size = (64, 64)

# Extract the patches
patches = image.extract_patches_2d(image, patch_size)

# convert from 4d array down to 3d array for better use
patches = patches.reshape(-1, patch_size[0], patch_size[1])

print(patches)
```

> imported our test images into this code segment
> 
> [[[174 201 231],[174 201 231]],[[173 200 230],[173 200 230]]] is the output I got after running the code

If we wanted to go the other way around (patches back to the image), *sklearn* has a library to do that as well. 

```
from sklearn.feature_extraction import image

# Load the patches
patches = ...

# Reconstruct the image
image = image.reconstruct_from_patches_2d(patches, (height, width))
```
