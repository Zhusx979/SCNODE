import torch

import torch
import torch.nn as nn
from torchdiffeq import odeint

'''
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, drop_rate=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size

        # 计算 patch 数量
        assert img_size % patch_size == 0, "图像尺寸必须能被 patch 尺寸整除"
        num_patches = (img_size // patch_size) ** 2

        # 每个 patch 展平后的维度
        patch_dim = 3 * patch_size * patch_size  # 3 是 RGB 通道数

        # 1. Patch Embedding: 将图像分块并映射到嵌入维度
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # 2. 位置嵌入 (Positional Embedding)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 是为了 CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # CLS token
        self.pos_drop = nn.Dropout(drop_rate)

        # 3. Transformer 编码器
        self.transformer = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate),
                "norm2": nn.LayerNorm(embed_dim),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(drop_rate),
                )
            })
            for _ in range(depth)
        ])

        # 4. 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]  # 批次大小

        # 1. 将图像分块并展平
        # x: (B, C, H, W) -> (B, num_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # torch.Size([1, 3, 14, 14, 16, 16])
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # torch.Size([1, 14, 14, 3, 16, 16])

        x = x.view(B, -1, self.patch_size * self.patch_size * 3)
        # torch.Size([1, 196, 768])
        # 2. Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 采用全连接层进行嵌入维度
        # 3. 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        # torch.Size([1, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        # torch.Size([1, 197, 768])

        # 4. 添加位置嵌入
        x = x + self.pos_embed
        # torch.Size([1, 197, 768])
        x = self.pos_drop(x)

        # 5. Transformer 编码器
        for block in self.transformer:
            # Multi-Head Attention
            x = block["norm1"](x)
            # torch.Size([1, 197, 768])
            attn_output, _ = block["attn"](x, x, x)
            # torch.Size([1, 197, 768])
            x = x + attn_output

            # MLP
            x = block["norm2"](x)
            mlp_output = block["mlp"](x)
            # torch.Size([1, 197, 768])
            x = x + mlp_output

        # 6. 分类
        x = self.norm(x)
        x = x[:, 0]  # 取 CLS token
        x = self.head(x)

        return x


# 示例：创建 ViT 模型
# model = VisionTransformer(
#     img_size=224,         # 输入图像尺寸
#     patch_size=16,        # 每个 patch 的大小
#     num_classes=1000,     # 分类数量（如 ImageNet 的 1000 类）
#     embed_dim=768,        # 嵌入维度
#     depth=12,             # Transformer 层数
#     num_heads=12,         # 注意力头数
#     mlp_ratio=4.0,        # MLP 隐藏层维度比例
#     drop_rate=0.1         # Dropout 率
# )

def Get_vit(num_classes=1000):
    """
    获取 ViT 模型实例
    :param num_classes: 分类类别数
    :param model_name: ViT 变种
    :param pretrained: 是否加载预训练权重
    :return: ViTClassifier 实例
    """
    return VisionTransformer(num_classes=num_classes)


# 创建一个 1x1x5x5 的特征图，内容为 1 到 25
# input_feature_map = torch.arange(1, 26).view(1, 1, 5, 5).float()
input_feature_map = torch.randn(1, 3, 224, 224)  # batch size = 1, 3 channels (RGB), 224x224 image
# 打印输入特征图
print("Input Feature Map:")
print(input_feature_map)

model = VisionTransformer(num_classes=3)
out = model(input_feature_map)

# 打印结果
print("\nProcessed Result:")
print(out.shape)
'''
class ODEFunc_att(nn.Module):
    def __init__(self, input_channels,hide_channels = 64):
        super(ODEFunc_att, self).__init__()
        # 使用简单的全连接层作为ODE的函数
        self.fc1 = nn.Linear(input_channels, hide_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hide_channels, input_channels)

    def forward(self, t, x):
        return self.fc2(self.relu(self.fc1(x)))


# ODE注意力机制
class ODEAttention(nn.Module):
    def __init__(self, input_channels, hide_channels = 64):
        super(ODEAttention, self).__init__()
        self.input_channels = input_channels

        # 定义ODE模型
        self.ode_func = ODEFunc_att(input_channels,hide_channels)

        # 平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出为单个值的池化

    def forward(self, x):
        # 获取输入的通道数
        batch_size, channels, height, width = x.size()

        # 1. 对输入进行平均池化（生成通道级别的全局特征）
        pooled_x = self.avg_pool(x)  # (batch_size, channels, 1, 1)
        pooled_x = pooled_x.view(batch_size, channels)  # 转换为 (batch_size, channels)

        # 2. 使用ODE模型生成连续时间的动态特征
        # 初始状态为池化后的特征
        t = torch.tensor([0., 1.], device=x.device)  # 时间从0到1
        ode_output = odeint(self.ode_func, pooled_x, t)  # 求解ODE

        # ODE的输出是一个大小为[2, batch_size, channels]的张量，取最终的输出
        ode_output = ode_output[1]  # (batch_size, channels)

        # 3. 将ODE输出的注意力与原始输入进行通道级别的加权
        ode_attention = ode_output.view(batch_size, channels, 1, 1)  # (batch_size, channels, 1, 1)
        x = x * ode_attention  # 按通道进行相乘，得到加权特征

        return x

# 示例用法
if __name__ == "__main__":
    input_feature_map = torch.randn(64,3,224,224)  # (batch_size=1, input_dim=12)
    # print("Input Feature Map:")
    # print(input_feature_map)

    import time
    start_time = time.time()  # 记录 epoch 开始时间
    model = ODEAttention(3)
    out = model(input_feature_map)  # 仅传入 x，不传入 t
    end_time = time.time()  # 记录 epoch 结束时间
    duration = end_time - start_time
    print(f'Time: {duration:.2f} sec')
    print("\nProcessed Result:")
    print(out.shape)