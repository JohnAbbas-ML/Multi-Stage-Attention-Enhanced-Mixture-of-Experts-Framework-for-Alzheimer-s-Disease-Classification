"""
Alzheimer Mixture of Experts (MoE) Model with Channel and Spatial Attention
Author: Muhammad John Abbas
"""

import torch
import torch.nn as nn
from torchsummary import summary  # install with: pip install torchsummary


# ---------------- Dense Block ----------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))

    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


# ---------------- Transition Layer ----------------
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


# ---------------- Channel Attention ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attention_weights = self.sigmoid(out)
        return attention_weights * x, attention_weights


# ---------------- Spatial Attention ----------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(out)
        attention_weights = self.sigmoid(attention_map)
        return attention_weights * x, attention_weights


# ---------------- Expert ----------------
class Expert(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, attention_type='channel'):
        super(Expert, self).__init__()
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
        out_channels = in_channels + growth_rate * num_layers
        self.attention_type = attention_type
        if attention_type == 'channel':
            self.attention = ChannelAttention(out_channels)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        x = self.dense_block(x)
        if self.attention_type in ['channel', 'spatial']:
            x, attention_weights = self.attention(x)
            return x, attention_weights
        else:
            return x, None


# ---------------- Gating Network ----------------
class GatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(GatingNetwork, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        pooled = self.pool(x)
        return self.gate(pooled)


# ---------------- MoE Stage ----------------
class MoEStage(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, num_experts=4, reduction_factor=2):
        super(MoEStage, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            attention_type = 'channel' if i % 2 == 0 else 'spatial'
            self.experts.append(Expert(in_channels, growth_rate, num_layers, attention_type))

        expert_output_channels = in_channels + growth_rate * num_layers
        self.gating = GatingNetwork(in_channels, num_experts)
        self.transition = TransitionLayer(expert_output_channels, expert_output_channels // reduction_factor)

    def forward(self, x):
        weights = self.gating(x)
        expert_outputs, attention_maps = [], []
        for expert in self.experts:
            output, attention = expert(x)
            expert_outputs.append(output)
            attention_maps.append(attention)

        combined = torch.zeros_like(expert_outputs[0])
        for i, output in enumerate(expert_outputs):
            combined += output * weights[:, i].view(-1, 1, 1, 1)

        out = self.transition(combined)
        return out, weights, attention_maps


# ---------------- Alzheimer MoE Model ----------------
class AlzheimerMoEModel(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, initial_channels=64):
        super(AlzheimerMoEModel, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = MoEStage(initial_channels, growth_rate=32, num_layers=6, num_experts=4)
        self.stage2 = MoEStage(initial_channels * 2, growth_rate=32, num_layers=12, num_experts=4)
        self.stage3 = MoEStage(initial_channels * 4, growth_rate=32, num_layers=24, num_experts=4)

        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(initial_channels * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(initial_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.initial_conv(x)
        x, gates1, attn1 = self.stage1(x)
        x, gates2, attn2 = self.stage2(x)
        aux_out = self.aux_classifier(x)
        x, gates3, attn3 = self.stage3(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        main_out = self.classifier(x)
        if return_attention:
            return main_out, aux_out, (gates1, gates2, gates3), (attn1, attn2, attn3)
        else:
            return main_out, aux_out, None, None


# ---------------- Run and Show Architecture ----------------
if __name__ == "__main__":
    model = AlzheimerMoEModel(input_channels=3, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\n===== Model Architecture =====\n")
    summary(model, (3, 224, 224))  # Show architecture summary

    print("\n===== Forward Pass Test =====\n")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    main_out, aux_out, _, _ = model(dummy_input)
    print(f"Main output shape: {main_out.shape}")
    print(f"Auxiliary output shape: {aux_out.shape}")
