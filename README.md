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
As aforementioned, I used 4 very high quality images of lungs with pneumonia. Here is one of them - they all look pretty similar from a third person perspective. 

<p align="center">
  <img 
    src="https://github.com/akhilvreddy/ECE491-SpecialProblems/blob/main/Training%20Images/im2_pn_normal.jpeg"
  >
</p>

If you look closely, the image has a black border on all sides. Since each pixel can inlfuence the model, we would like to remove the border. This is what I did to remove the border from the image: 

``` 
import cv2
# Load the image
image = cv2.imread("image.jpg")

# Define the border color (black)
lower = [0, 0, 0]
upper = [0, 0, 0]

# Create a mask for the border color
mask = cv2.inRange(image, lower, upper)

# Remove the border color
image_without_border = cv2.bitwise_not(image, image, mask=mask)
```

Another way to do this is using the properies of python. 

``` 
Image1 = img[:, 1:]
Image1 = Image1[:, :-1]
```

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

## Tackling the problem

### Autoencoders 
Autoencoders can be used to remove the noise and distortions from the images by training the network to reconstruct the original, noise-free image from the noisy input image. Autoencoders don't have a specific defintion but they are capable of reducing the data dimensions by ignoring noise in the data. It will then expand the data out agian to the dimensions of the initial dataset. There are usually four components inside an autoencoder, all of these combined make it up.
- Encoder  
  * In which the model learns how to reduce the input dimensions and compress the input data into an encoded representation.
- Bottleneck 
   * The layer that contains the compressed representation of the input data. This is the lowest possible dimensions of the input data.
- Decoder
  * In which the model learns how to reconstruct the data from the encoded representation to be as close to the original input as possible.
- Reconstruction Loss
  * the method that measures measure how well the decoder is performing and how close the output is to the original input.

Here is an image depicting the way autoencoders work. This image shows a one-layer design, but a lot of autoencoders can have many more than a single layer.

<p align="center">
  <img 
    width="464"
    height="328"
    src="https://github.com/akhilvreddy/ECE491/blob/main/ourimage.png"
  >
</p>

Autoencoders are the biggest tools that allow us to solve inverse problems. The way we are going to solve the vector equation is by trying to inverse it, kind of like an algebraic equation, but we cannot do the same elementary operations for a vector equation involving matrices. 

### Recovery Algorithms

Recovery Algorithms are basically a set of instructions or steps that are used to restore or recover data that has been lost, deleted, or damaged. In this case, our images are going to be hit with speckle noise and we would want to recover the original, clean image from this. One of the ways to get the signal (or image in this case) back from speckle noise is by Projected Gradient Descent. The cost function in this case would be 

#### Recovery using Generative Functions (GFs)

Recovery using generative functions refers to the process of using generative models to recover or generate data that has been lost, corrupted, or never existed in the first place. Generative models are a class of machine learning models that are trained to generate new data that is similar to the training data.

#### Projected Gradient Descent (PGD)

This is one of the ways we reduce the cost function step-by-step to drive closer to the solution everytime. Understading the pseudo-code is very important before coding it up. 

* We are assuming that the result is **x<sub>t</sub>**.

```
Xo = diag(xo)    //initalize the matrix
Bo = A(Xo)^2(A^T)

for t = 1:T do 
  for i = 1:n do
  
  s(t, i) = *some algorithmic step*
  
  end 
  
  x_t = pi*c_r*s_t
  Xt = diag(xt)
  Bt = A(X_t)^2(A^T)
end
```

Let's unpack this alogrithm. The first two lines should be pretty self-explanatory - we are just setting up the basic matrix and constants in order to do the first iteration calcuations.

The nested for loops is where we get into the meat of the algorithm. For the inner-most loop, we are trying to find values of s_(t,i). Since t stays the same for a single loop, we get n specific values of the s vector. 

After exiting that loop, we set all the values of the x_t vector to a specific constant times those values. 

The X_t matrix gets updated from this everytime. 

By running through both of these for-loops we are inching closer towards an answer everytime. The main line that is getting us to reduce our cost-function is: 

<p align="center">
  <img 
    width="508"
    height="48"
    src="https://github.com/akhilvreddy/ECE491/blob/main/pic2.png"
  >
</p>

## Starting with the Autoencoder class

### Dependencies 
As you can see in the main jupyter notebook, we have a couple of dependencies for this program. 
```
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import numpy as np
from matplotlib import pyplot as plt
import cv2
```

- Scikit-learn (sklearn) is a popular Python library for machine learning. It offers a wide range of tools for tasks such as classification, regression, clustering, and model selection, and it is built on top of other popular Python libraries such as NumPy and Matplotlib. It provides a consistent interface to various machine learning models, making it easy to switch between them and to perform common tasks such as feature extraction, model selection, and evaluation. I have used it through the program, especially in my autoencoder class. 
- NumPy is a Python library that provides support for large multi-dimensional arrays and matrices of numerical data, as well as a large collection of mathematical functions to operate on these arrays. It is a fundamental package for scientific computing with Python. 
- OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision. It provides a number of features such as image processing, video analysis, object detection, and machine learning.
- Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK. It provides a high-level interface for drawing attractive and informative statistical graphics.

All of these libraries together help us get to our final goal - a fully working autoencoder method which stores the *MSE (mean squared error)*. 

### Putting our Images into Python
As aforementioned, we start by importing our image into python from our desktop. We need to divide by 255 because we want to normalize it. After normalization, we output the image and as you can see, it looks pretty much the same as what we had above. Thsi is because if you divide every pixel value by the same value, you are going to get the same image, just scaled down in value. 
```
# inputting the image from 
input_img = "im1_pn_normal.jpeg"

#saving the images that we have into vector variables
img = cv2.imread(input_img,0)

# the following command will help us understand what the image will look like (vectorized)
img = img/255
# this is going to show us the dimensions of the image  (we can make adjustments based off this)
```

We can now output this image using the following command:
```
plt.imshow(img,cmap='gray')
```
<p align="center">
  <img 
    src="https://github.com/akhilvreddy/ECE491-SpecialProblems/blob/main/Reference%20Images/img1_matplotliboutput.png"
  >
</p>

### Getting the patches 
As aforementioned, we wanted to use patch images for certain capacity reasons. Here is how we do it in the program.
```
k = 8
N = 10000
patch = image.extract_patches_2d(img, (k, k), max_patches = N, random_state=None) # change the max_patches number as well as random_state # change around the random_state value
patch.shape
```

> (10000, 8, 8) is the output (patches size)

Out of all those patches we would want to see one of them, and to make sure that they look pretty much the same throughout, we would want 
```
import random
plt.imshow(255*patch[random.randint(0,N)],cmap='gray')
```

> This command really shows us the power of *matplotlib* in python.

<p align="center">
  <img 
    src="https://github.com/akhilvreddy/ECE491-SpecialProblems/blob/main/Reference%20Images/img2_firstpatch.png"
  >
</p>

> We do not know exactly where and which part of the image this patch comes from, but we can definitely feel that this is a patch from the image above.

### More dependencies 
Moving on to the real Machine Learning part, we would want to import *PyTorch* so that we can use it's ability to make NNs.
```
import torch
from torchvision import datasets
from torchvision import transforms
```

### Patches in Numpy
Above, we have the patches from our image. Here, we would want to convert it to a numpy patch tensor so that we can do calculations and manipulations with it. Especially since we are going to be feeding these into our autoencoder. 
```
patchtensor = torch.from_numpy(patch)
print(patchtensor.data.shape)
type(patchtensor)
```

> torch.Size([10000, 8, 8]) (notice how size stays the same)
> torch.Tensor (torch.Tensor is the type now)

With everything settled, we would want to now ready the data (patches) for training. 
```
# DataLoader is used to load the dataset for training
patchloader = torch.utils.data.DataLoader(dataset = patchtensor, batch_size = 32, shuffle = True)
```

### Choosing our Autoencoder size 

This is probably the most difficult and important step of the whole process. Since choosing the size and dimensions have the most direct impact on the quality and efficacy of the autoencoder. I'll outline some of the sizes I have chosen. 

#### The first attempt:
256 (16 x 16) ==> 196 (14 x 14) ==> 144 (12 x 12) ==> 100 (10 x 10) ==> 64 (8 x 8) ==> 36 (6 x 6) ==> 25 (5 x 5)


25 (5 x 5) ==> 36 (6 x 6) ==> 64 (8 x 8) ==> 100 (10 x 10) ==> 144 (12 x 12) ==> 196 (14 x 14) ==> 256 (16 x 16)
  
These were the dimensions I used for the first time I did the autoenocder class and this brought in not the best results. The issue that happend with this is that the MSE was not as good as we wanted it to be, it was kind of all over the place. Here is what it looked like: 

<p align="center">
  <img 
    src="https://github.com/akhilvreddy/ECE491-SpecialProblems/blob/main/Reference%20Images/imgw_firstmse.png"
  >
</p>

As you can see, this is not a great reduction in MSE so we can do better. 

#### The second attempt: 
1024 (32 x 32) ==> 625 (25 x 25) ==> 400 (20 x 20) ==> 225 (15 x 15) ==> 144 (12 x 12) ==> 121 (11 x 11) ==> 100 (10 x 10)


100 (10 x 10) ==> 121 (11 x 11) ==> 144 (12 x 12) ==> 225 (15 x 15) ==> 400 (20 x 20) ==> 625 (25 x 25) ==> 1024 (32 x 32)

This was better, but still not the best. Here is a snippet of how this is inputted into python: 

```
torch.nn.Linear(k * k, 2000), 
torch.nn.ReLU(),
torch.nn.Linear(2000, 1000),
torch.nn.ReLU(),
torch.nn.Linear(1000, 500),
torch.nn.ReLU(),
torch.nn.Linear(500, 200),
torch.nn.ReLU(),
torch.nn.Linear(200, 100),
 ```
 
Each Linear command is followed by a ReLU command. The ReLU activation function is commonly used in neural networks to introduce non-linearity and improve the model's ability to learn complex, non-linear relationships in the data and hence having linear commands followed by the ReLU commands help the model really understand the data.

#### The third attempt: 
After not much of a decrease in MSE, I decided there were other ways to fix the issues that I was having. Going to the bottle-neck need not to be uniform and I used this fact to my ability.

1024 (32 x 32) ==> 625 (25 x 25) ==> 400 (20 x 20) ==> 225 (15 x 15) ==> 144 (12 x 12) ==> 121 (11 x 11) ==> 100 (10 x 10)


100 (10 x 10) ==> 121 (11 x 11) ==> 144 (12 x 12) ==> 225 (15 x 15) ==> 400 (20 x 20) ==> 625 (25 x 25) ==> 1024 (32 x 32)

Here is the MSE we got by doing the above: 

### Building the end to end Autoencoder class 
```
# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28 # change these values
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9

        #grow first and then shrink
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(k * k, 2000),  # change these values, these are not big enough ## change the 32^2 to maybe 2048
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
        )
    
        '''
        what can we do with the compressed form of the nn?
        can we take this nn and put it somewhere else so that it can work as transfer of data with much less information
        '''

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, k * k),
            torch.nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

```

### Using our model 

```
# Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001, weight_decay = 1e-8)
```
What is happening here is that we are making an instance of our Autoencoder class, kind of like in an OOP language like java. After that, we set up our MSE calculations in a variable called *loss_function* which comes from PyTorch's MSELoss() class. 

This line of code creates an optimizer object, which will be used to update the parameters of a model during training. The optimizer used here is the Adam optimizer, which is a popular choice for training neural networks. The Adam optimizer uses a combination of gradient descent and adaptive learning rate techniques to adjust the model's parameters.

The first argument passed to the Adam function is the parameters of the model, so the optimizer will update the parameters of the model object. The second argument is the learning rate (lr), which is set to 0.0001. This value determines the step size at which the optimizer makes updates to the model's parameters. A smaller learning rate means that the optimizer will make smaller updates, while a larger learning rate means that the optimizer will make larger updates.

The third argument is weight_decay, which is set to 1e-8. It helps to prevent overfitting by adding an L2 penalty on the weights of the model during optimization.

Learning rate of 0.0001 and weight decay of 1e-8 are reasonable default values to start with for most problems.

### Training the Autoencoder model to our specific data
This is the most important part of the whole project - it is everything coming together and training the AE model so that we can get reconstructed images. 
```
epochs = 300 #change the epoch value to be larger
outputs = []
losses = []
for epoch in range(epochs):
#     print(epoch)
    for image in patchloader:
        image = image.reshape(-1, k*k)# Reshaping the image to (-1, 784)
        image = image.float()

    # Output of Autoencoder
        reconstructed = model(image)

    # Calculating the loss function
        loss = loss_function(reconstructed, image)

    # The gradients are set to zero,
    # the gradient is computed and stored.
    # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Storing the losses in a list for plotting
        losses.append(loss)
        outputs.append((epochs, image, reconstructed))
    print('epoch [{}/{}], loss:{:.8f}'
          .format(epoch + 1, epochs, loss.data.detach().numpy()))
```
The outermost loop is running for a total of 300 epochs, which is the value assigned to the variable epochs. An epoch is a full training pass through all the training data.

The innermost loop is iterating through a data loader object called patchloader, which is presumably an object that loads the images that will be used as input to the autoencoder. For each image in the patchloader, the code reshapes the image to a 2D tensor of shape (-1, k*k) and converts the image to a float data type.

The autoencoder model is then applied to the image, and the output is assigned to the variable reconstructed. The loss function is then calculated by comparing the reconstructed image to the original image. The loss function used here is the mean squared error.

The gradients are then set to zero, the gradient is computed and stored by the backward() method, and the step() method updates the model parameters using the optimizer.

At each iteration of the inner loop, the loss is appended to the losses list and the output and reconstructed image are appended to the outputs list.

Finally, the code prints out the current epoch number, the total number of epochs, and the value of the loss at that point in the training.

### MSE Outputs
Here is the output that we get from the model running above. It is worth to note that this took my machine 17 and a half minutes to run. This is important to think about for scalability. If it takes a decent laptop running i5 17.5 minutes, to train huge AI models, it would require a lot of CPU and GPU processing power which would cost a lot. 
```
epoch [1/300], loss:0.06043266
epoch [2/300], loss:0.07363702
epoch [3/300], loss:0.06667658
epoch [4/300], loss:0.06946521
epoch [5/300], loss:0.05480300
epoch [6/300], loss:0.09153187
epoch [7/300], loss:0.09710239
epoch [8/300], loss:0.07143620
epoch [9/300], loss:0.05849411
...
epoch [291/300], loss:0.00412072
epoch [292/300], loss:0.00621012
epoch [293/300], loss:0.00533406
epoch [294/300], loss:0.00514319
epoch [295/300], loss:0.00585328
epoch [296/300], loss:0.00399428
epoch [297/300], loss:0.00497398
epoch [298/300], loss:0.00542437
epoch [299/300], loss:0.00350400
epoch [300/300], loss:0.00568663
```
This is the output of the training loop, which shows the value of the loss function at the end of each epoch. The loss function is a measure of how well the autoencoder is able to reconstruct the input images. A lower loss value indicates that the autoencoder is performing well and is able to accurately reconstruct the input images.

It is clear from the output that the loss is decreasing as the training progresses. In the beginning, the loss is high around 0.06 and by the end, it is around 0.0055. And this decrease in loss indicates that the model is learning and improving with each epoch.

It is also worth noting that the loss fluctuates, which is normal. There will be some variations in the loss values between different epochs, but overall, the loss is decreasing as the training progresses.

### Viewing our MSE 
```
l = []
for j in range(len(losses)):
    a = losses[j].detach().numpy()
    l.append(a)

# Defining the Plot Style
plt.plot(l)
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
```

This code is plotting the training loss over the course of the training process.

The first loop iterates through the losses list, which contains the loss values at the end of each iteration of the training loop. For each value in the losses list, the code converts the value to a numpy array and appends it to a new list l.

The next part of the code is using the matplotlib library to plot the data in the l list. The plt.plot(l) function plots the values in the l list on the y-axis, and the x-axis is determined by the index of the values in the list.

This plot will show how the loss changes over time and will help to monitor the training progress and identify if the model is overfitting or underfitting.

Here is what a good version of what a loss model would look like: 

### Viewing our reconstructed image: 
```
# print(reconstructed.shape)
for i, item in enumerate(reconstructed):
    item = item.reshape(-1, k, k)
    plt.imshow(item[0].detach().numpy(),cmap='gray',vmin=0, vmax=1)
#     plt.show()
```

Here is what one the reconstructed patches look like:

We can also stitch them together using the method we talked about [above](https://github.com/akhilvreddy/ECE491-SpecialProblems/blob/main/README.md#test-patches). Here is what the code would look like: 
```
image = image.reconstruct_from_patches_2d(patches, (height, width))
```

## Final Thoughts

In conclusion, Autoencoders are a powerful tool for unsupervised feature learning and dimensionality reduction. They can be used in a wide range of applications, such as image and speech recognition, natural language processing, and anomaly detection.

In this research paper, we have discussed the basic principles of autoencoders and how they work. We have also demonstrated how to implement an autoencoder in Pytorch, a popular deep learning framework. We have trained the autoencoder on a dataset of images and analyzed the results. The loss function decreased as the training progressed, and the output of the autoencoder was able to reconstruct the input images with high accuracy.

We have also discussed some of the limitations of autoencoders, such as overfitting and the need for a large amount of data. However, these limitations can be overcome by using techniques such as regularization, denoising, and variational autoencoders.

In summary, autoencoders are a versatile and powerful tool for machine learning and have many potential applications. Further research in this field could lead to new and improved methods for unsupervised feature learning and dimensionality reduction.

I would like to thank Professor Jalali and my TA Mengyu Zhao for helping me and adivising me through this project.



