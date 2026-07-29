import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 残差网络中的basicblock
class Bottleneck(nn.Module):
    expansion = 4      # 输出通道数的倍乘
   #因为Bottleneck 每个block里面三个卷积层通道数是不一样的，最后一个是前两个的4倍，所以expansion=4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



"""一般的ResNet50"""
class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  # layers=参数列表
        # 对于stage0，大家都是一样的
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


def Get_ResNet50(num_classes):
    return ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)



"""SE-ResNet50"""
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=2):  # layers=参数列表
        # 对于stage0，大家都是一样的
        self.inplanes = 64
        super(SE_ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.channel_attention = ChannelAttention(64)
        self.channel_attention1 = ChannelAttention(256)
        self.channel_attention2 = ChannelAttention(512)
        self.channel_attention3 = ChannelAttention(1024)
        self.channel_attention4 = ChannelAttention(2048)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.channel_attention(x)
        x = self.layer1(x)
        x = self.channel_attention1(x)
        x = self.layer2(x)
        x = self.channel_attention2(x)
        x = self.layer3(x)
        x = self.channel_attention3(x)
        x = self.layer4(x)
        x = self.channel_attention4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x

def Get_SE_ResNet50(num_classes):
    return SE_ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)



"""LVPN-ResNet50"""
def Level_vertical_pooling(x):
    L_inf = torch.max(torch.sum(torch.abs(x), dim=3), dim=2).values.unsqueeze(2)
    L1 = torch.max(torch.sum(torch.abs(x), dim=2), dim=2).values.unsqueeze(2)
    feature_cat_vec = torch.cat((L_inf, L1), dim=2).flatten(1)
    return feature_cat_vec

class LVP_ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(LVP_ChannelAttention, self).__init__()
        self.LVP = Level_vertical_pooling
        self.fc1 = nn.Linear(2 * in_planes, int(1.5 * in_planes))
        self.relu1 = nn.Mish()
        self.fc2 = nn.Linear(int(1.5 * in_planes), in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x
        x = self.LVP(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x).unsqueeze(2).unsqueeze(3)
        x = self.sigmoid(x) * tmp
        return x

class LVPN_ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=2):  # layers=参数列表
        # 对于stage0，大家都是一样的
        self.inplanes = 64
        super(LVPN_ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.channel_attention = LVP_ChannelAttention(64)
        self.channel_attention1 = LVP_ChannelAttention(256)
        self.channel_attention2 = LVP_ChannelAttention(512)
        self.channel_attention3 = LVP_ChannelAttention(1024)
        self.channel_attention4 = LVP_ChannelAttention(2048)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.channel_attention(x)
        x = self.layer1(x)
        x = self.channel_attention1(x)
        x = self.layer2(x)
        x = self.channel_attention2(x)
        x = self.layer3(x)
        x = self.channel_attention3(x)
        x = self.layer4(x)
        x = self.channel_attention4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


def Get_LVPN_ResNet50(num_classes):
    return LVPN_ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)














'''WBCA'''

def WHSP_Pooling(feature_maps):
    if feature_maps.size(2) <= 2 and feature_maps.size(3) <= 2:
        # 提取整个特征图的最大值（保留批量和通道维度）
        max_val = torch.max(feature_maps.flatten(start_dim=2), dim=2).values
        return max_val.unsqueeze(-1)  # 保持维度一致 [batch, channel, 1]

    top_max = torch.max(feature_maps[:, :, 0, 0:-1], dim=2).values.unsqueeze(2) # 顶部边界
    # print(top_max)
    left_max = torch.max(feature_maps[:, :, 1:, 0], dim=2).values.unsqueeze(2) # 左侧边界
    # print(left_max)
    bottom_max = torch.max(feature_maps[:, :, -1, 1:], dim=2).values.unsqueeze(2) # 底部边界
    # print(bottom_max)
    right_max = torch.max(feature_maps[:, :, :-1, -1], dim=2).values.unsqueeze(2)# 右侧边界
    # print(right_max)


    # 将边界最大值合并到一起
    result = torch.cat((top_max, left_max, bottom_max, right_max), dim=2)
    # print(result)


    # 提取内部矩阵并递归处理
    inner_matrix = feature_maps[:, :, 1:-1, 1:-1]
    # print(inner_matrix)
    inner_result = WHSP_Pooling(inner_matrix)
    # print(inner_result)
    # 拼接边界和内部矩阵的结果
    result = torch.cat((result, inner_result), dim=2)
    # print(result.shape)

    return result

def L2(feature_map):
    l2_vector = torch.norm(feature_map, p=2, dim=2)
    # print(l2_vector)
    return l2_vector

class WHSP_ChannelAttention(nn.Module):
    def __init__(self, in_channels,reduction_ratio=2):
        super(WHSP_ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.WHSP = WHSP_Pooling  # 假设 WHSP_Pooling 是预定义的池化模块
        self.l2 = L2

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),  # 缩减通道数
            nn.Mish(),
            nn.Linear(in_channels // reduction_ratio, in_channels)  # 恢复通道数
        )

        self.relu1 = nn.Mish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x  # 保存原始输入

        # Step 1: 通过 WHSP 池化获取特征
        pooled = self.WHSP(x)  # 假设输出形状为 [batch, pooled_features]
        l2 = L2(pooled)
        x = self.mlp(l2)
        # x = x + l2
        # 调整维度以匹配原始输入
        x = x.view(x.size(0), self.in_channels, 1, 1)  # 形状 [batch, in_planes, 1, 1]

        # Step 4: 应用注意力权重
        return self.sigmoid(x) * tmp

class WHSP_ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=2):  # layers=参数列表
        # 对于stage0，大家都是一样的
        self.inplanes = 64
        super(WHSP_ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.channel_attention = WHSP_ChannelAttention(64)
        self.channel_attention1 = WHSP_ChannelAttention(256)
        self.channel_attention2 = WHSP_ChannelAttention(512)
        self.channel_attention3 = WHSP_ChannelAttention(1024)
        self.channel_attention4 = WHSP_ChannelAttention(2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.channel_attention(x)
        x = self.layer1(x)
        x = self.channel_attention1(x)
        x = self.layer2(x)
        x = self.channel_attention2(x)
        x = self.layer3(x)
        x = self.channel_attention3(x)
        x = self.layer4(x)
        x = self.channel_attention4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


def Get_WHSP_ResNet50(num_classes):
    return WHSP_ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4, device=None):
        super().__init__()
        self.S = S
        self.device = device  # 存储设备信息

        # 定义卷积层
        self.convs = nn.ModuleList()
        for i in range(S):
            conv = nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            self.convs.append(conv.to(device))  # 将卷积层移动到设备上

        # 定义SE模块
        self.se_blocks = nn.ModuleList()
        for i in range(S):
            se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            self.se_blocks.append(se.to(device))  # 将SE模块移动到设备上

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.shape)
        # Step1: SPC Module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs, s, ci, h, w
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :].to(self.device))  # 确保卷积操作在相同设备上

        # Step2: SE Weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :].to(self.device)))  # 确保SE模块在相同设备上
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3: Softmax
        softmax_out = self.softmax(SE_out)

        # Step4: SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out



# ResNet50 with PSA
class PSA_ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  # layers=参数列表
        # 对于stage0，大家都是一样的
        self.inplanes = 64
        super(PSA_ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.channel_attention = PSA(64)
        self.channel_attention1 = PSA(256)
        self.channel_attention2 = PSA(512)
        self.channel_attention3 = PSA(1024)
        self.channel_attention4 = PSA(2048)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.channel_attention(x)
        x = self.layer1(x)
        x = self.channel_attention1(x)
        x = self.layer2(x)
        x = self.channel_attention2(x)
        x = self.layer3(x)
        x = self.channel_attention3(x)
        x = self.layer4(x)
        x = self.channel_attention4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


def Get_PSA_ResNet50(num_classes):
    return PSA_ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)