import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import os
from torchvision import transforms
from torch.nn import DataParallel

# all these imports are working the 

os.environ["CUDA_VISIBLE_DEVICES"]='0'
ids=[0]
torch.cuda.empty_cache()

#the same values I used in the other file
EPOCH = 50
BATCH_SIZE = 64
LR = 0.001

# load images data
train_data = torchvision.datasets.ImageFolder('C:\Users\reddy\Desktop\TrainingImages', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class UNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

unet=DataParallel(UNet(in_ch=1,out_ch=1)).cuda()
optimizer=torch.optim.Adam(unet.parameters(),lr=LR) 
loss_func=nn.MSELoss() #loss function: MSE

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder3(nn.Module):
    def __init__(self):
        super(Autoencoder3, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder4(nn.Module):
    def __init__(self):
        super(Autoencoder4, self).__init__()
        self.encoder = nn.Sequential(  # two layers encoder
            nn.Linear(74*74, 8000),
            nn.ReLU(True),  # ReLU, Tanh, etc.
            nn.Linear(8000, 1000),
            nn.ReLU(True),  # input is in (0,1), so select this one
        )
        self.decoder = nn.Sequential(  # two layers decoder
            nn.Linear(1000, 8000),
            nn.ReLU(True),
            nn.Linear(8000, 74*74),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Load 4 NNs
autoencoder1=Autoencoder1()
autoencoder1.load_state_dict(torch.load('4NNs-part1.pkl', map_location=lambda storage, loc:storage)) 
autoencoder2=Autoencoder2()
autoencoder2.load_state_dict(torch.load('4NNs-part2.pkl', map_location=lambda storage, loc:storage))
autoencoder3=Autoencoder3()
autoencoder3.load_state_dict(torch.load('4NNs-part3.pkl', map_location=lambda storage, loc:storage))
autoencoder4=Autoencoder4()
autoencoder4.load_state_dict(torch.load('4NNs-part4.pkl', map_location=lambda storage, loc:storage))

for epoch in range(EPOCH):
    for step, (x,x_label) in enumerate(train_loader): 
        b_x=x.view(-1,128*128)
        b_y=x.cuda()
        batch_size = x.size()[0]
        b_xx = b_x.view(batch_size, 128, 128)

        b_x1 = torch.narrow(torch.narrow(b_xx, 2, 0, 74), 1, 0, 74).contiguous().view(-1, 74 * 74).cuda()
        b_x2 = torch.narrow(torch.narrow(b_xx, 2, 54, 74), 1, 0, 74).contiguous().view(-1, 74 * 74).cuda()
        b_x3 = torch.narrow(torch.narrow(b_xx, 2, 0, 74), 1, 54, 74).contiguous().view(-1, 74 * 74).cuda()
        b_x4 = torch.narrow(torch.narrow(b_xx, 2, 54, 74), 1, 54, 74).contiguous().view(-1, 74 * 74).cuda()
        
        # running
        decoded1 = autoencoder1(b_x1).view(batch_size, 74, 74)
        decoded2 = autoencoder2(b_x2).view(batch_size, 74, 74)
        decoded3 = autoencoder3(b_x3).view(batch_size, 74, 74)
        decoded4 = autoencoder4(b_x4).view(batch_size, 74, 74)

        # concatenation
        decoded1_se = torch.narrow(decoded1, 2, 0, 54)
        decoded1_co = torch.narrow(decoded1, 2, 54, 20)
        decoded2_se = torch.narrow(decoded2, 2, 20, 54)
        decoded2_co = torch.narrow(decoded2, 2, 0, 20)
        decoded3_se = torch.narrow(decoded3, 2, 0, 54)
        decoded3_co = torch.narrow(decoded3, 2, 54, 20)
        decoded4_se = torch.narrow(decoded4, 2, 20, 54)
        decoded4_co = torch.narrow(decoded4, 2, 0, 20)
        decoded12_ave = (decoded1_co + decoded2_co) / 2
        decoded34_ave = (decoded3_co + decoded4_co) / 2
        up_part = torch.cat([decoded1_se, decoded12_ave, decoded2_se], dim=2)
        down_part = torch.cat([decoded3_se, decoded34_ave, decoded4_se], dim=2)
        up_part_se = torch.narrow(up_part, 1, 0, 54)
        up_part_co = torch.narrow(up_part, 1, 54, 20)
        down_part_se = torch.narrow(down_part, 1, 20, 54)
        down_part_co = torch.narrow(down_part, 1, 0, 20)
        updown_ave = (up_part_co + down_part_co) / 2
        decoded = torch.cat([up_part_se, updown_ave, down_part_se], dim=1).cuda()
        
        #running neural network
        output=unet(decoded)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50==0:
            print('Epoch:',epoch, '| tran loss : %.4f' % loss.data.cpu().numpy())

torch.save(unet.module.state_dict(),'Unet-refine-Xray.pkl') #saving our code as a package