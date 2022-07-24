from torch import nn
import torch
from torch.nn import functional as F
import cv2

PATH = "./resnet34-b627a593.pth"


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)


class ResNet(nn.Module):
    '''
    實現主Module: ResNet34
    ResNet34包含多個layer，每個layer又包含多個Residual block
    用子Module來實現Residual Block，用make_layer函數來實現layer
    '''

    def __init__(self):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 分類的Layer，分別有3, 4, 6個Residual Block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分類用的Fully Connection
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        構建Layer，包含多個Residual Block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # o = F.avg_pool2d(x4, 7)
        # o = o.view(o.size(0), -1)
        # return self.fc(o)
        return x1, x2, x3, x4


class ConvG(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size1=3, kernel_size2=3):
        super(ConvG, self).__init__()
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size1, stride, kernel_size1 // 2)
        self.Conv2 = nn.Conv2d(in_channel, out_channel, kernel_size2, stride, kernel_size2 // 2)

    def forward(self, x):
        out1 = self.Conv1(x)
        out2 = self.Conv2(x)
        out = out1 + out2
        return F.relu(out)


class ConvG_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernel_size1=3, kernel_size2=3):
        super(ConvG_Block, self).__init__()
        self.ConvG = ConvG(in_channel, out_channel, 2, kernel_size1, kernel_size2)
        self.ConvNxN = nn.Sequential(
            nn.Conv2d(out_channel, 512, kernel_size1, stride, kernel_size1 // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.ConvG(x)
        out2 = self.ConvNxN(out1)
        return out1, out2


class AFM(nn.Module):
    def __init__(self):
        super(AFM, self).__init__()

        self.HighBranch = nn.Sequential(
            nn.MaxPool3d((4, 1, 1)),
            nn.Conv2d(128, 512, 1),
            nn.ReLU(inplace=True)
        )

        self.MidBranch = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )

        self.ConvG_Block1 = ConvG_Block(512, 128, 4, 7, 3)
        self.ConvG_Block2 = ConvG_Block(128, 128, 2, 5, 3)
        self.ConvG_Block3 = ConvG_Block(128, 128, 1, 3, 3)

    def forward(self, x):
        high = self.HighBranch(x)
        mid = self.MidBranch(x)
        low1, low4 = self.ConvG_Block1(x)
        low2, low5 = self.ConvG_Block2(low1)
        low3, low6 = self.ConvG_Block3(low2)

        attention_weight = low4 + low5 + low6
        attention_weight = F.sigmoid(attention_weight)
        attention_weight = attention_weight.repeat([1, 1, 8, 8])

        attention = attention_weight.mul(mid)
        return attention + high


class CAM(nn.Module):
    def __init__(self, num_channel, pooling_kernel_size):
        super(CAM, self).__init__()
        self.pooling_kernel_size = pooling_kernel_size
        self.AvgP = nn.AvgPool3d((1, pooling_kernel_size, pooling_kernel_size))
        self.MaxP = nn.MaxPool3d((1, pooling_kernel_size, pooling_kernel_size))
        self.MLP = DoubleConv(num_channel, num_channel)

    def forward(self, x):
        avg_out = self.AvgP(x)
        max_out = self.MaxP(x)
        mlp_out = self.MLP(avg_out) + self.MLP(max_out)
        attention_weight = mlp_out.repeat([1, 1, self.pooling_kernel_size, self.pooling_kernel_size])
        attention_weight = F.sigmoid(attention_weight)
        attention = attention_weight.mul(x)
        return attention


class SAM(nn.Module):
    def __init__(self, num_channel, pooling_kernel_size):
        super(SAM, self).__init__()
        self.num_channel = num_channel
        self.pooling_kernel_size = pooling_kernel_size
        self.AvgP = nn.AvgPool3d((pooling_kernel_size, 1, 1))
        self.MaxP = nn.MaxPool3d((pooling_kernel_size, 1, 1))
        self.Conv = DoubleConv(num_channel * 2 // pooling_kernel_size, 2, mid_channels=num_channel)

    def forward(self, x):
        avg_out = self.AvgP(x)
        max_out = self.MaxP(x)
        cat_out = torch.cat((avg_out, max_out), 1)
        conv_out = self.Conv(cat_out)
        conv_out = F.sigmoid(conv_out)
        attention_weight = conv_out.repeat([1, self.num_channel // 2, 1, 1])
        attention = attention_weight.mul(x)
        return attention


class ADM_Block(nn.Module):
    def __init__(self, in_channel, out_channel, channel_kernel_size, space_kernel_size):
        super(ADM_Block, self).__init__()

        self.cam = CAM(in_channel, channel_kernel_size)
        self.sam = SAM(in_channel, space_kernel_size)

        self.up = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2)
        self.DoubleConv = DoubleConv(in_channel, out_channel)

    def forward(self, low_feature, high_feature):
        fusion = low_feature + high_feature
        attention = self.cam(fusion) + self.sam(fusion)
        up_out = self.up(attention)
        out = self.DoubleConv(up_out)
        return out


class ADM(nn.Module):
    def __init__(self):
        super(ADM, self).__init__()
        self.adm1 = ADM_Block(128, 64, 32, 2)
        self.adm2 = ADM_Block(256, 128, 16, 2)
        self.adm3 = ADM_Block(512, 256, 8, 2)
        self.adm4 = ADM_Block(512, 512, 4, 2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, res1, res2, res3, res4, afm):
        adm4_out = self.adm4(res4, afm)
        adm3_out = self.adm3(res3, adm4_out)
        adm2_out = self.adm2(res2, adm3_out)
        adm1_out = self.adm1(res1, adm2_out)
        out = self.up(adm1_out)
        return out


class ACNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(ACNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoder = ResNet()
        self.attention = AFM()
        self.decoder = ADM()

    def forward(self, images):
        x1, x2, x3, x4 = self.encoder(images)
        att = self.attention(x4)
        result = self.decoder(x1, x2, x3, x4, att)
        return result

    def load_resnet_weights(self, PATH, device):
        self.encoder.load_state_dict(torch.load(PATH, map_location=device), strict=False)


if __name__ == "__main__":
    model = ACNet()
    model.load_resnet_weights(PATH)

    input = torch.autograd.Variable(torch.randn(1, 3, 2560, 1440))
    output = model(input)

    segmentation = output.detach().numpy().transpose([0, 2, 3, 1])[0]
    segmentation[segmentation > 0.5] = 255
    segmentation[segmentation < 0.5] = 0
    print(segmentation.shape)

    # cv2.imshow("", segmentation)
    # cv2.waitKey(0)
    cv2.imwrite("./seg.jpg", segmentation)



