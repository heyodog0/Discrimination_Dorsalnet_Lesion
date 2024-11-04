import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from resblocks import ResBlock, BottleneckTransform


class SymmetricConv3d(nn.Module):
    """Convolution, adding symmetric versions for equivariance."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, weight_norm=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_norm = weight_norm

        k = 1 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        w = 2 * np.sqrt(k) * (torch.rand(out_channels, in_channels, *kernel_size) - 0.5)
        w = nn.Parameter(w)
        self.register_parameter("weight", w)

    def forward(self, X):
        w = torch.cat(
            (
                torch.rot90(self.weight, 0, [3, 4]),
                torch.rot90(self.weight, 1, [3, 4]),
                torch.rot90(self.weight, 2, [3, 4]),
                torch.rot90(self.weight, 3, [3, 4]),
            ),
            axis=0,
        )

        if self.weight_norm:
            return F.conv3d(
                X,
                w
                / torch.sqrt(
                    (w ** 2)
                    .sum(1, keepdims=True)
                    .sum(2, keepdims=True)
                    .sum(3, keepdims=True)
                ),
                padding=self.padding,
                stride=self.stride,
            )
        else:
            return F.conv3d(X, w, padding=self.padding, stride=self.stride)


class ShallowNet(nn.Module):
    """
    Create a tiny shallow network to approximate the whole freakin brain.
    """

    def __init__(
        self, nstartfeats=16, symmetric=False, dropout_rate=0.1, weight_norm=False
    ):
        super().__init__()

        if symmetric:
            assert nstartfeats % 4 == 0
            self.conv1 = SymmetricConv3d(
                3,
                nstartfeats // 4,
                [5, 7, 7],
                [1, 2, 2],
                padding=[2, 3, 3],
                weight_norm=weight_norm,
            )
        else:
            self.conv1 = nn.Conv3d(
                3, nstartfeats, [5, 7, 7], [1, 2, 2], padding=[2, 3, 3], bias=False
            )

            if weight_norm:
                self.conv1 = nn.utils.weight_norm(self.conv1, "weight")

        self.bn1 = nn.BatchNorm3d(nstartfeats)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

        self.dropout = nn.Dropout3d(dropout_rate)
        self._dropout_rate = dropout_rate

        self.layers = [
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("relu", self.relu),
            ("pool", self.pool_layer),
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool_layer(x)

        if self._dropout_rate > 0:
            x = self.dropout(x)

        # Endpoint
        return x


class DorsalNet(nn.Module):
    def __init__(self, symmetric=True, nfeats=32):
        super().__init__()
        self.s1 = ShallowNet(
            nstartfeats=nfeats * 2,
            symmetric=symmetric,
            dropout_rate=0,
            weight_norm=False,
        )

        self.res0 = ResBlock(
            nfeats * 2, nfeats, 1, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res1 = ResBlock(
            nfeats, nfeats, 3, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res2 = ResBlock(
            nfeats, nfeats, 1, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res3 = ResBlock(
            nfeats, nfeats, 3, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.dropout = nn.Dropout3d(0.1)
        self.concat = Identity()

        # Hack to get visualization working properly.
        self.layers = [
            ("conv1", self.s1.conv1),
            ("bn1", self.s1.bn1),
            ("res0", self.res0),
            ("res1", self.res1),
            ("res2", self.res2),
            ("res3", self.res3),
            ("concat", self.concat),
        ]

        self.conv1 = self.s1.conv1  # Visualize weights

    def forward(self, x):
        x0 = self.s1(x)
        x1 = self.res0(x0)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)

        # Add two types of features together
        self.concat(torch.cat((x0, x4), dim=1))

        x = self.dropout(x4)

        return x
    

class ShallowNet_Alt(nn.Module):
    """
    Create a tiny shallow network to approximate the whole freakin brain.
    """

    def __init__(
        self, nstartfeats=16, symmetric=False, dropout_rate=0.1, weight_norm=False
    ):
        super().__init__()

        if symmetric:
            assert nstartfeats % 4 == 0
            self.conv1 = SymmetricConv3d(
                3,
                nstartfeats // 4,
                [5, 7, 7],
                [1, 2, 2],
                padding=[2, 3, 3],
                weight_norm=weight_norm,
            )
        else:
            self.conv1 = nn.Conv3d(
                3, nstartfeats, [5, 7, 7], [1, 2, 2], padding=[2, 3, 3], bias=False
            )

            if weight_norm:
                self.conv1 = nn.utils.weight_norm(self.conv1, "weight")

        self.conv2 = nn.Conv3d(3, 5, [7, 11, 11], [1, 2, 2], padding=[3, 5, 5])
        self.bn1 = nn.BatchNorm3d(nstartfeats)
        #self.fc = nn.Linear(nstartfeats,nstartfeats)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

        self.dropout = nn.Dropout3d(dropout_rate)
        self._dropout_rate = dropout_rate

        self.layers = [
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            #("fc", self.fc),
            ("relu", self.relu),
            ("pool", self.pool_layer),
            ("conv2", self.conv2),
        ]

    def forward(self, x):
        x1= self.conv1(x)
        x1 = self.bn1(x1)

        x2 = self.conv2(x)
        combined = torch.cat((x1, x2), dim=1)

        x = self.relu(combined)
        x = self.pool_layer(x)
        

        if self._dropout_rate > 0:
            x = self.dropout(x)

        # Endpoint
        return x


class DorsalNet_Alt(nn.Module):
    def __init__(self, symmetric=True, nfeats=32):
        super().__init__()
        self.s1 = ShallowNet_Alt(
            nstartfeats=nfeats * 2,
            symmetric=symmetric,
            dropout_rate=0,
            weight_norm=False,
        )

        self.res0 = ResBlock(
            nfeats * 2 + 5, nfeats, 1, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res1 = ResBlock(
            nfeats, nfeats, 3, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res2 = ResBlock(
            nfeats, nfeats, 1, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res3 = ResBlock(
            nfeats, nfeats, 3, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.dropout = nn.Dropout3d(0.1)
        self.concat = Identity()

        # Hack to get visualization working properly.
        self.layers = [
            ("conv1", self.s1.conv1),
            ("bn1", self.s1.bn1),
            ("res0", self.res0),
            ("res1", self.res1),
            ("res2", self.res2),
            ("res3", self.res3),
            ("concat", self.concat),
            ("conv2", self.s1.conv2)
        ]

        self.conv1 = self.s1.conv1  # Visualize weights

    def forward(self, x):
        x0 = self.s1(x)
        x1 = self.res0(x0)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)

        # Add two types of features together
        self.concat(torch.cat((x0, x4), dim=1))

        x = self.dropout(x4)

        return x

class DoG3d(nn.Module):
    """
    Create a convolutional layer that applies a difference of gaussians to the input.
    The trainable parameters are the weights and variaces of the two gaussians
    plus a exponential temporal decay factor.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if kernel_size[1] == kernel_size[2]:
            self.kernel_size = kernel_size[2]
        else:
            raise ValueError("Kernel size must be the same in the spatial dimensions")
        self.temporal_size = kernel_size[0]
        
        self.variance1 = nn.Parameter(torch.ones(out_channels)*2.0)
        self.variance2 = nn.Parameter(torch.ones(out_channels)*1.0)
        self.weight1 = nn.Parameter(torch.ones(out_channels)*1.0)
        self.weight2 = nn.Parameter(torch.ones(out_channels)*1.0)

        tal0 = torch.tensor(3.66)
        tal1 = torch.tensor(7.16)
        K = lambda t: torch.pow(t,6) / torch.pow(tal0,7) * torch.exp(-t/tal0) - torch.pow(t,6) / torch.pow(tal1,7) * torch.exp(-t/tal1)
        times = torch.flip(torch.arange(20,20+30*self.temporal_size,30).float(),[0])
        self.temp_kernel = K(times)

        self.weights=self.compute_dog_weights()

        

    def compute_dog_weights(self):
        device = self.variance1.device
        weights = torch.zeros(self.out_channels, self.in_channels, self.temporal_size, self.kernel_size, self.kernel_size).to(device)
        for i in range(self.out_channels):
            l = self.kernel_size//2
            x,y = torch.meshgrid(torch.arange(-l,l+1).to(device), torch.arange(-l,l+1).to(device))
            x = x.repeat(self.temporal_size,1,1).float()
            y = y.repeat(self.temporal_size,1,1).float()

            temp_kernel = self.temp_kernel.unsqueeze(1).unsqueeze(1).to(device)

            gaussian1 = self.weight1[i]*torch.exp(-0.5*(x**2 + y**2)/(self.variance1[i]**2))*temp_kernel
            gaussian2 = self.weight2[i]*torch.exp(-0.5*(x**2 + y**2)/(self.variance2[i]**2))*temp_kernel

            dog = gaussian1 - gaussian2
            for j in range(self.in_channels):
                weights[i,j,:,:,:] = dog

        return weights
    
    def forward(self, X):
        self.weights= self.compute_dog_weights()
        padding = (self.temporal_size//2, self.kernel_size//2, self.kernel_size//2)
        return nn.functional.conv3d(X, self.weights, padding=padding)


class ShallowNet_DoG(nn.Module):
    """
    Create a tiny shallow network to approximate the whole freakin brain.
    """

    def __init__(
        self, nstartfeats=16, symmetric=False, dropout_rate=0.1, weight_norm=False
    ):
        super().__init__()

        if symmetric:
            assert nstartfeats % 4 == 0
            self.conv1 = SymmetricConv3d(
                5,
                nstartfeats // 4,
                [5, 7, 7],
                [1, 2, 2],
                padding=[2, 3, 3],
                weight_norm=weight_norm,
            )
        else:
            self.conv1 = nn.Conv3d(
                5, nstartfeats, [5, 7, 7], [1, 2, 2], padding=[2, 3, 3], bias=False
            )

            if weight_norm:
                self.conv1 = nn.utils.weight_norm(self.conv1, "weight")
        
        self.dog = DoG3d(3, 5, [3, 7, 7])
        self.dog_pool = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.bn1 = nn.BatchNorm3d(nstartfeats+5)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool_layer = nn.MaxPool3d(
            kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
        )

        self.dropout = nn.Dropout3d(dropout_rate)
        self._dropout_rate = dropout_rate

        self.layers = [
            ("dog", self.dog),
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("relu", self.relu),
            ("pool", self.pool_layer),
        ]

    def forward(self, x):
        x = self.dog(x)
        x2 = self.dog_pool(x)
        x1 = self.conv1(x)
        combined = torch.cat((x1, x2), dim=1)

        x = self.bn1(combined)
        x = self.relu(combined)
        x = self.pool_layer(x)

        if self._dropout_rate > 0:
            x = self.dropout(x)

        # Endpoint
        return x
    
class DorsalNet_DoG(nn.Module):
    def __init__(self, symmetric=True, nfeats=32):
        super().__init__()
        self.s1 = ShallowNet_DoG(
            nstartfeats=nfeats * 2,
            symmetric=symmetric,
            dropout_rate=0,
            weight_norm=False,
        )

        self.res0 = ResBlock(
            nfeats * 2 + 5, nfeats, 1, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res1 = ResBlock(
            nfeats, nfeats, 3, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res2 = ResBlock(
            nfeats, nfeats, 1, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.res3 = ResBlock(
            nfeats, nfeats, 3, 1, BottleneckTransform, 8, drop_connect_rate=0.2
        )

        self.dropout = nn.Dropout3d(0.1)
        self.concat = Identity()

        # Hack to get visualization working properly.
        self.layers = [
            ("dog", self.s1.dog),
            ("conv1", self.s1.conv1),
            ("bn1", self.s1.bn1),
            ("res0", self.res0),
            ("res1", self.res1),
            ("res2", self.res2),
            ("res3", self.res3),
            ("concat", self.concat),
        ]

        self.conv1 = self.s1.conv1  # Visualize weights

    def forward(self, x):
        x0 = self.s1(x)
        x1 = self.res0(x0)
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)

        # Add two types of features together
        self.concat(torch.cat((x0, x4), dim=1))

        x = self.dropout(x4)

        return x
    

class ShallowDorsalNet(nn.Module):
    def __init__(self, symmetric=True, nfeats=32):
        super().__init__()
        self.s1 = ShallowNet(
            nstartfeats=nfeats * 2,
            symmetric=symmetric,
            dropout_rate=0,
            weight_norm=False,
        )

        self.res0 = ResBlock(
            nfeats * 2,
            nfeats,
            1,
            1,
            BottleneckTransform,
            nfeats // 2,
            drop_connect_rate=0.2,
        )

        self.res1 = ResBlock(
            nfeats,
            nfeats,
            3,
            1,
            BottleneckTransform,
            nfeats // 2,
            drop_connect_rate=0.2,
        )

        self.dropout = nn.Dropout3d(0.1)

        # Hack to get visualization working properly.
        self.layers = [
            ("conv1", self.s1.conv1),
            ("bn1", self.s1.bn1),
            ("res0", self.res0),
            ("res1", self.res1),
        ]

        self.conv1 = self.s1.conv1  # Visualize weights

    def forward(self, x):
        x0 = self.s1(x)
        x1 = self.res0(x0)
        x2 = self.res1(x1)

        x = self.dropout(x2)

        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
