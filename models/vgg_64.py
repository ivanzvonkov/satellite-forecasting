import torch
import torch.nn as nn
from torch.autograd import Variable

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                vgg_layer(64, 64),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 16 x 16 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):   # [50, 1, 64, 64]
        h1 = self.c1(input) # 64 -> 32                  [50, 64, 64, 64]
        h2 = self.c2(self.mp(h1)) # 32 -> 16            [50, 128, 32, 32]
        h3 = self.c3(self.mp(h2)) # 16 -> 8             [50, 256, 16, 16]
        h4 = self.c4(self.mp(h3)) # 8 -> 4              [50, 512, 8, 8]
        h5 = self.c5(self.mp(h4)) # 4 -> 1              [50, 90, 1, 1]
        return h5.view(-1, self.dim), [h1, h2, h3, h4]  # [50, 90], [...]


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256*2, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2, 128),
                vgg_layer(128, 64)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64*2, 64),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        # [50, 90], [([50, 64, 64, 64]), ([50, 128, 32, 32]), ([50, 256, 16, 16]), ([50, 512, 8, 8])]
        vec, skip = input 

        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4           [50, 512, 4, 4]
        up1 = self.up(d1) # 4 -> 8                                      [50, 512, 8, 8] = h4.shape

        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 8 x 8            [50, 256, 8, 8]
        up2 = self.up(d2) # 8 -> 16                                     [50, 256, 16, 16] = h3.shape

        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 16 x 16          [50, 128, 16, 16]
        up3 = self.up(d3) # 8 -> 32                                     [50, 128, 32, 32] = h2.shape

        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 32 x 32          [50, 64, 32, 32]
        up4 = self.up(d4) # 32 -> 64                                    [50, 64, 64, 64] = h1.shape

        output = self.upc5(torch.cat([up4, skip[0]], 1)) # 64 x 64      [50, 1, 64, 64]
        return output

class gaussian_encoder(nn.Module):
    def __init__(self, dim, output_size, nc=1):
        super(gaussian_encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                vgg_layer(64, 64),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 16 x 16 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.output_size = output_size
        self.mu_net = nn.Linear(dim, output_size)
        self.logvar_net = nn.Linear(dim, output_size)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        h5 = self.c5(self.mp(h4)) # 4 -> 1
        mu = self.mu_net(h5.view(-1, self.dim))
        logvar = self.logvar_net(h5.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, [h1, h2, h3, h4]
        # return  h5.view(-1, self.dim),