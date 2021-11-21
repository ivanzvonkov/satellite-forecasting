import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class encoder(nn.Module):
  def __init__(self, dim, nc=1):
    super(encoder,self).__init__()
    self.dim = dim
    nf = 64
    
    self.conv1=nn.Conv2d(nc,nf,5,padding=2,stride=2)   #in_channels=3
    self.bn1=nn.BatchNorm2d(nf,momentum=0.9)

    self.conv2=nn.Conv2d(nf,nf * 4,5,padding=2,stride=2)
    self.bn2=nn.BatchNorm2d(nf * 4,momentum=0.9)

    self.relu=nn.LeakyReLU(0.2)

    self.fc1=nn.Linear(256*16*16, 2048)
    self.bn3=nn.BatchNorm1d(2048,momentum=0.9)

    self.fc_mean=nn.Linear(2048,dim)
    self.fc_logvar=nn.Linear(2048,dim)
  
  def reparameterize(self, mean, logvar):
    std = logvar.mul(0.5).exp_()
        
    #sampling epsilon from normal distribution
    epsilon=Variable(torch.randn(mean.shape[0],self.dim)).to(device)
    z = mean+std*epsilon
    return z

  def forward(self,x):
    batch_size=x.size()[0]
    h1=self.relu(self.bn1(self.conv1(x))) # [50, 1, 64, 64] -> [50, 64, 32, 32]
    h2=self.relu(self.bn2(self.conv2(h1))) # [50, 64, 32, 32] -> [50, 256, 16, 16]

    out=h2.view(batch_size,-1) # [50, 65536]
    h3=self.relu(self.bn3(self.fc1(out))) # [50, 2048]

    mean=self.fc_mean(h3)
    logvar=self.fc_logvar(h3)
    z = self.reparameterize(mean, logvar) # [50, 90]

    return z.view(-1, self.dim), [h1, h2, h3]  # [50, 90], [...]

class decoder(nn.Module):
  def __init__(self, dim, nc=1):
    super(decoder,self).__init__()
    self.dim = dim
    nf = 64

    self.fc1 = nn.Linear(dim, 2048)
    self.bn1 = nn.BatchNorm1d(2048, momentum=0.9)

    self.fc2=nn.Linear(2048 * 2,16*16*256)
    self.bn2=nn.BatchNorm1d(16*16*256,momentum=0.9)

    self.relu=nn.LeakyReLU(0.2)

    self.deconv1=nn.ConvTranspose2d(nf*4*2,nf,6, stride=2, padding=2)
    self.bn3=nn.BatchNorm2d(nf,momentum=0.9)

    self.deconv2=nn.ConvTranspose2d(nf*2,1,6, stride=2, padding=2)
    self.tanh=nn.Tanh()

  def reparameterize(self, mean, logvar):
    std = logvar.mul(0.5).exp_()
        
    #sampling epsilon from normal distribution
    epsilon=Variable(torch.randn(mean.shape[0],self.dim)).to(device)
    z = mean+std*epsilon
    return z

  def forward(self,x):    
    z, (h1, h2, h3) = x                               # [50, 90], [...]
    fc1 = self.relu(self.bn1(self.fc1(z)))            # [50, 2048] = h3.shape

    concat1 = torch.cat([fc1, h3], 1)
    d1=self.relu(self.bn2(self.fc2(concat1)))         # [50, 256*16*16]

    d1=d1.view(-1,256,16,16)                          # [50, 256, 16, 16] = h2.shape

    concat2 = torch.cat([d1, h2], 1)
    d2=self.relu(self.bn3(self.deconv1(concat2)))     # [50, 64, 32, 32] = h1.shape

    concat3 = torch.cat([d2, h1], 1)
    d3=self.tanh(self.deconv2(concat3))               # [50, 1, 64, 64]

    # x = self.reparameterize(x[0], x[1])
    # x=self.relu(self.bn1(self.fc1(x)))
    # x=x.view(-1,256,16,16)                  # [50, 256, 16, 16]
    # x=self.relu(self.bn2(self.deconv1(x)))  # [50, 64, 32, 32]
    # x=self.relu(self.bn4(self.deconv3(x)))  # [50, 32, 64, 64]
    # x=self.tanh(self.deconv4(x))            # [50, 1, 64, 64]
    return d3
    