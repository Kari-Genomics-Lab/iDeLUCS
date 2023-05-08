import torch 
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super (block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity =  x.clone()

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1 = self.make_layer(block, layers[0], 16, 1)
        self.layer_2 = self.make_layer(block, layers[1], 32, 2)
        self.in_channels = 32
        self.layer_3 = self.make_layer(block, layers[2], 64, 2)
        self.in_channels = 64
        self.layer_4 = self.make_layer(block, layers[3], 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return self.softmax(x), x  


    def make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, 
                                                          intermediate_channels, 
                                                          kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(intermediate_channels))
                                                

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(intermediate_channels, intermediate_channels))

        return nn.Sequential(*layers)



class ResNet_k_5(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet_k_5,self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer_1 = self.make_layer(block, layers[0], 16, 1)
        self.layer_2 = self.make_layer(block, layers[1], 32, 2)
        self.in_channels = 32
        self.layer_3 = self.make_layer(block, layers[2], 64, 2)
        self.in_channels = 64
        self.layer_4 = self.make_layer(block, layers[3], 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return self.softmax(x), x  


    def make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, 
                                                          intermediate_channels, 
                                                          kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(intermediate_channels))
                                                

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(intermediate_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet18(img_channel=3, num_classes=50):
    return ResNet(block, [2,2,2,2], img_channel, num_classes)

def ResNet18_k_5(img_channel=3, num_classes=50):
    return ResNet_k_5(block, [2,2,2,2], img_channel, num_classes)


class ConvNet( nn.Module):
    def __init__(self, img_channel = 1, num_classes=10):
        super(ConvNet,self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
                nn.Conv2d(img_channel, 16, 5, 1, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AvgPool2d(3,2,1),
                nn.Conv2d(16,32,3,1,1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(3,2,1),
                nn.Conv2d(32,64,3,1,1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3,2,1),
                nn.Conv2d(64,64,3,1,1),
                nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        latent = self.layers(x)
        latent = latent.reshape(latent.shape[0], -1)
        out = self.softmax(self.fc(latent))

        return out, latent 


class ConvNet_k_5( nn.Module):
    def __init__(self, img_channel = 1, num_classes=10):
        super(ConvNet_k_5,self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
                nn.Conv2d(img_channel, 16, 5, 1, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16,32,3,1,1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(3,2,1),
                nn.Conv2d(32,64,3,1,1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3,2,1),
                nn.Conv2d(64,64,3,1,1),
                nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        latent = self.layers(x)
        latent = latent.reshape(latent.shape[0], -1)
        out = self.softmax(self.fc(latent))

        return out, latent 


def test(device):
    net = ResNet18_k_5(img_channel=1)
    x = torch.rand(2,1, 64, 64)

    #net = ConvNet(img_channel=1)
    #x = torch.rand(2,1, 64, 64)


    print(net)
    net.to(device)
    y = net(x)
    print(y[0].shape)

#test('cpu')

