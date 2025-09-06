"""
Mamba-MFDAM (多尺度特征融合域适应模块) 实现
专为YOLOv8训练阶段的域适应设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class GradientReversalLayer(Function):
    """梯度反转层，用于对抗训练"""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播时反转梯度
        return -ctx.alpha * grad_output, None


class GRL(nn.Module):
    """梯度反转层包装器"""

    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

    def set_alpha(self, alpha):
        """设置梯度反转强度参数"""
        self.alpha = alpha


class MambaBlock(nn.Module):
    """简化的Mamba块，用于特征处理"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model  # 模型维度
        self.d_state = d_state  # 状态维度
        self.d_conv = d_conv  # 卷积核大小
        self.expand = expand  # 扩展因子
        self.d_inner = int(self.expand * self.d_model)  # 内部维度

        # 输入投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,  # 分组卷积
            padding=d_conv - 1,
        )

        # 状态空间模型参数
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # 输出投影层
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 激活函数
        self.act = nn.SiLU()

    def forward(self, x):
        """
        前向传播
        x: (B, L, D) 其中 B=批次大小, L=序列长度, D=特征维度
        """
        B, L, D = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # 分割为两部分: (B, L, d_inner)

        # 卷积处理 (需要转置以适应conv1d)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)

        # 状态空间模型计算 (简化版本)
        dt = self.dt_proj(x_conv)  # (B, L, d_inner)
        dt = F.softplus(dt)

        # 简化的状态空间计算
        x_ssm = x_conv * dt  # 简化版本

        # 门控和输出
        z = self.act(z)
        output = x_ssm * z

        # 输出投影
        output = self.out_proj(output)

        return output


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块，用于MFFDC"""

    def __init__(self, channels_list, out_channels=256):
        super(MultiScaleFeatureFusion, self).__init__()
        self.channels_list = channels_list  # 输入通道数列表
        self.out_channels = out_channels  # 输出通道数

        # 特征适配层，将不同通道数的特征统一到相同维度
        self.adaptations = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),  # 1x1卷积调整通道数
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in channels_list
        ])

        # 注意力机制用于特征融合
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * len(channels_list), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(channels_list), 1),
            nn.Sigmoid()  # 生成权重
        )

        # Mamba处理模块
        self.mamba = MambaBlock(out_channels)

    def forward(self, features):
        """
        前向传播
        features: 来自不同尺度的特征图列表
        """
        # 将所有特征适配到相同的通道维度
        adapted_features = []
        target_size = features[0].shape[2:]  # 使用第一个特征的空间尺寸作为目标

        for i, feat in enumerate(features):
            adapted = self.adaptations[i](feat)
            # 如果空间尺寸不匹配，进行插值调整
            if adapted.shape[2:] != target_size:
                adapted = F.interpolate(adapted, size=target_size, mode='bilinear', align_corners=False)
            adapted_features.append(adapted)

        # 拼接特征用于注意力计算
        concat_features = torch.cat(adapted_features, dim=1)
        attention_weights = self.attention(concat_features)  # (B, num_scales, H, W)

        # 应用注意力权重
        fused_feature = torch.zeros_like(adapted_features[0])
        for i, feat in enumerate(adapted_features):
            weight = attention_weights[:, i:i + 1, :, :]  # (B, 1, H, W)
            fused_feature += feat * weight

        # 应用Mamba处理
        B, C, H, W = fused_feature.shape
        # 重塑为Mamba输入格式: (B, H*W, C)
        fused_flat = fused_feature.view(B, C, H * W).transpose(1, 2)
        mamba_out = self.mamba(fused_flat)
        # 重塑回原始格式: (B, C, H, W)
        fused_feature = mamba_out.transpose(1, 2).view(B, C, H, W)

        return fused_feature


class MFFDC(nn.Module):
    """多尺度特征融合域分类器"""

    def __init__(self, channels_list, num_classes=2, fusion_channels=256):
        super(MFFDC, self).__init__()
        self.fusion = MultiScaleFeatureFusion(channels_list, fusion_channels)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 域分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_channels, fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_channels // 2, num_classes)
        )

    def forward(self, features):
        # 融合多尺度特征
        fused = self.fusion(features)

        # 全局池化
        pooled = self.global_pool(fused).flatten(1)  # (B, C)

        # 域分类
        domain_pred = self.classifier(pooled)

        return domain_pred


class FocalLoss(nn.Module):
    """Focal Loss，用于处理类别不平衡问题"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡因子
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction  # 损失减少方式

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算概率
        pt = torch.exp(-ce_loss)
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MambaMFDAM(nn.Module):
    """完整的Mamba-MFDAM模块"""

    def __init__(self, channels_list, num_domains=2, alpha_init=1.0):
        super(MambaMFDAM, self).__init__()
        self.grl = GRL(alpha_init)  # 梯度反转层
        self.mffdc = MFFDC(channels_list, num_domains)  # 多尺度特征融合域分类器
        self.focal_loss = FocalLoss(alpha=1, gamma=2)  # Focal损失
        self.training_only = True  # 仅在训练时使用

    def forward(self, features, domain_labels=None):
        """
        前向传播
        features: 来自YOLOv8 neck的特征图列表
        domain_labels: 训练时的域标签 (推理时为None)
        """
        if not self.training or not self.training_only:
            return None  # 推理时不进行前向传播

        # 应用梯度反转
        reversed_features = [self.grl(feat) for feat in features]

        # 域分类
        domain_pred = self.mffdc(reversed_features)

        # 如果提供了标签，计算损失
        if domain_labels is not None:
            domain_loss = self.focal_loss(domain_pred, domain_labels)
            return domain_pred, domain_loss

        return domain_pred, None

    def set_alpha(self, alpha):
        """更新GRL的alpha参数"""
        self.grl.set_alpha(alpha)