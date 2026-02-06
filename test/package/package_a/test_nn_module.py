# Owner(s): ["oncall: package/deploy"]

import smith


class TestNnModule(smith.nn.Module):
    def __init__(self, nz=6, ngf=9, nc=3):
        super().__init__()
        self.main = smith.nn.Sequential(
            # input is Z, going into a convolution
            smith.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            smith.nn.BatchNorm2d(ngf * 8),
            smith.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            smith.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            smith.nn.BatchNorm2d(ngf * 4),
            smith.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            smith.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            smith.nn.BatchNorm2d(ngf * 2),
            smith.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            smith.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            smith.nn.BatchNorm2d(ngf),
            smith.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            smith.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            smith.nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
