"""
Attention Mechanisms for Object Detection
============================================
Implements CBAM (Convolutional Block Attention Module) from
'CBAM: Convolutional Block Attention Module' (Woo et al., ECCV 2018).

CBAM applies channel attention followed by spatial attention,
allowing the network to focus on "what" (channels) and "where"
(spatial locations) are most informative.

Architecture:
    Input → ChannelAttention → SpatialAttention → Output
    F' = Mc(F) ⊗ F        (channel-refined features)
    F'' = Ms(F') ⊗ F'     (spatially-refined features)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module (SE-style squeeze-excitation).

    Learns inter-channel dependencies by compressing spatial dims
    via global pooling, then applying a shared MLP bottleneck.

    Architecture:
        Input(C,H,W) → AvgPool + MaxPool → SharedMLP → Sigmoid → Scale

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio for the MLP bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        mid = max(channels // reduction, 1)

        # Shared MLP: two FC layers with ReLU in between
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Channel-attended tensor (B, C, H, W).
        """
        B, C, H, W = x.shape

        # Global Average Pooling → (B, C)
        avg_out = x.mean(dim=[2, 3])
        avg_out = self.mlp(avg_out)

        # Global Max Pooling → (B, C)
        max_out = x.amax(dim=[2, 3])
        max_out = self.mlp(max_out)

        # Combine both pooling paths
        attention = torch.sigmoid(avg_out + max_out)  # (B, C)

        # Reshape and apply: (B, C) → (B, C, 1, 1)
        return x * attention.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial Attention Module.

    Learns where to attend by applying a conv layer on pooled
    channel descriptors (avg + max along channel axis).

    Architecture:
        Input(C,H,W) → ChannelPool → Conv7x7 → Sigmoid → Scale

    Args:
        kernel_size: Convolution kernel size (default 7).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        padding = kernel_size // 2

        # 2-channel input (avg + max pooled) → 1-channel attention map
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Spatially-attended tensor (B, C, H, W).
        """
        # Pool across channel dimension → (B, 1, H, W) each
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)

        # Concatenate → (B, 2, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)

        # Conv → Sigmoid → (B, 1, H, W)
        attention = torch.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies channel attention then spatial attention.
    This dual attention refines features by both channel and spatial
    dimensions, improving detection of small objects like license plates.

    Reference:
        Woo et al., 'CBAM: Convolutional Block Attention Module', ECCV 2018
        https://arxiv.org/abs/1807.06521

    Args:
        channels: Number of input feature channels.
        reduction: Channel attention reduction ratio.
        kernel_size: Spatial attention conv kernel size.

    Example:
        >>> cbam = CBAM(channels=256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = cbam(x)  # shape: (1, 256, 32, 32)
    """

    def __init__(self, channels: int, reduction: int = 16,
                 kernel_size: int = 7):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM: channel attention → spatial attention.

        Args:
            x: Input feature map (B, C, H, W).

        Returns:
            Attention-refined feature map (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (Hu et al., CVPR 2018).

    Adaptively recalibrates channel-wise feature responses by
    modelling inter-channel dependencies.

    Architecture:
        Input → GlobalAvgPool → FC → ReLU → FC → Sigmoid → Scale

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        mid = max(channels // reduction, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Recalibrated tensor (B, C, H, W).
        """
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)       # (B, C)
        scale = self.excitation(scale)             # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)
