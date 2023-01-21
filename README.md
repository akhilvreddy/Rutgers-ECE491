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

It is caused by the constructive and destructive interference of light waves scattered by small, randomly distributed scatterers within the imaging system. The scattered waves combine to create a speckled pattern on the image. There can be different variations of noise disturbances such as multiplicative noise and additive nosie.

Speckle noise can be a significant problem in certain imaging applications, as it can reduce image quality, make it difficult to detect small features, and limit the ability to extract useful information from the image. Getting rid of it can be difficult but can prove to be very useful. As the thesis mentions, it can "[lead] to more accurate diagnosis and treatment of the disease".
