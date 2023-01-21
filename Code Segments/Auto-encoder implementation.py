import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
from torchvision import transforms

# all these imports are working

#initial values - the changes I did are based on the size of the images I have 
EPOCH = 50
BATCH_SIZE = 64 # change this value since I don't have as many number of images
LR = 0.0005
N_TEST_IMG = 4

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ids = [0]
torch.cuda.empty_cache()

# load images data - you have to lead the vectorized image by using ToTensor - added a lot more of those images to that folder on desktop
train_data = torchvision.datasets.ImageFolder('C:\Users\reddy\Desktop\TrainingImages', transform=transforms.Compose([transforms.ToTensor(),]))

# change this part to the patches that I have gotten, see how to do that

train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Single NN network
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential( #two layers encoder
            nn.Linear(64*64,8000),
            nn.ReLU(True), #ReLU, Tanh, etc.
            nn.Linear(8000,3000),
            nn.ReLU(True) #input is in (0,1), so select this one
        )
        self.decoder=nn.Sequential( #two layers decoder
            nn.Linear(3000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 64*64),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded, decoded

autoencoder=Autoencoder().cuda()
optimizer=torch.optim.Adam(autoencoder.parameters(),lr=LR) # I don't see what this part is yet
loss_func=nn.MSELoss() #loss function: MSE

for epoch in range(EPOCH):
    for step, (x,x_label) in enumerate(train_loader): #train_loader has the number of batches, data, and label
        b_x=x.view(-1,64*64).cuda() #input data
        b_y=x.view(-1,64*64).cuda() #comparing data
        
        #running in the neural network
        encoded, decoded=autoencoder(b_x)
        loss=loss_func(decoded,b_y)
        optimizer.zero_grad() #initialize the optimizer
        loss.backward()
        optimizer.step()

        if step%10==0:
            print('Epoch:',epoch, '| tran loss : %.4f' % loss.data.cpu().numpy(),'count:', len(b_x[1,:]))

torch.save(autoencoder.state_dict(),'NNSingular.pkl') #save the parameter values of neural network