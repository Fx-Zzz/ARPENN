import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out = torch.max(x, dim=2, keepdim=True)[0]
        max_out = torch.max(max_out, dim=3, keepdim=True)[0]
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)
        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        self.res_block2 = ResidualBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.res_block3 = ResidualBlock(256, 256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.res_block4 = ResidualBlock(512, 512)
        self.dropout = nn.Dropout(0.1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc_output1 = nn.Linear(64, 1)  # For first layer output
        self.fc_output2 = nn.Linear(128, 1)  # For second layer output
        self.fc_output3 = nn.Linear(256, 1)  # For third layer output
        self.fc_output4 = nn.Linear(512, 1)  # For fourth layer output

    def forward(self, x):
        layers_outputs = []  # 用于存储每层的输出
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.res_block1(x)
        layers_outputs.append(self.fc_output1(self.avg_pool(x).flatten(1,-1)))  # Save first layer output
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.res_block2(x)
        layers_outputs.append(self.fc_output2(self.avg_pool(x).flatten(1,-1)))  # Save second layer output
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.res_block3(x)
        layers_outputs.append(self.fc_output3(self.avg_pool(x).flatten(1,-1)))  # Save third layer output
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.res_block4(x)
        layers_outputs.append(self.fc_output4(self.avg_pool(x).flatten(1,-1)))  # Save fourth layer output
        x = self.dropout(x)
        x = self.avg_pool(x).flatten(1,-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        print(x.shape)
        return x, layers_outputs  # Return final output and all layer outputs

if __name__ == "__main__":
    model = CNN(num_classes=1)  # 创建模型实例
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
