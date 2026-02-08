import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==========================================
# 1. SHARED FEATURE EXTRACTORS (The "Backbones")
# ==========================================

class AudioEncoder(nn.Module):
    """
    Standard 2D CNN for Spectrograms.
    Input: [Batch, 1, 64, 94] (for ~3s audio)
    Output: [Batch, 128]
    """
    def __init__(self):
        super().__init__()
        # 4 Conv blocks to downsample the spectrogram
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        
        # Adaptive pool ensures output size is fixed regardless of slight input variations
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x # Shape: [Batch, 128]

class VideoEncoder(nn.Module):
    """
    Pretrained ResNet-18 for Video Frames.
    Input: [Batch, Frames, 3, 224, 224]
    Output: [Batch, Frames, 512]
    """
    def __init__(self, freeze_layers=True):
        super().__init__()
        # Load standard ResNet
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        
        # Remove the final classification layer (fc)
        # We want the features before classification (512 dim)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers if requested (speeds up training)
        if freeze_layers:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x shape: [Batch, Frames, 3, H, W]
        batch, frames, c, h, w = x.shape
        
        # Merge Batch and Frames dimensions to pass through ResNet
        # [Batch*Frames, 3, H, W]
        x_reshaped = x.view(batch * frames, c, h, w)
        
        # Extract features: [Batch*Frames, 512, 1, 1]
        features = self.features(x_reshaped)
        
        # Flatten: [Batch*Frames, 512]
        features = features.view(batch * frames, -1)
        
        # Reshape back to separate frames: [Batch, Frames, 512]
        features = features.view(batch, frames, -1)
        
        return features

# ==========================================
# 2. ATTENTION MECHANISM (From previous step)
# ==========================================

class CrossModalAttention(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(audio_dim, hidden_dim)
        self.key_proj = nn.Linear(video_dim, hidden_dim)
        self.value_proj = nn.Linear(video_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, audio_features, video_features):
        # Audio (Query): [Batch, Audio_Dim] -> [Batch, 1, Hidden]
        query = self.query_proj(audio_features).unsqueeze(1)
        
        # Video (Key/Value): [Batch, Frames, Video_Dim] -> [Batch, Frames, Hidden]
        key = self.key_proj(video_features)
        value = self.value_proj(video_features)
        
        # Attention: Query looks at Key/Value
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        
        # Squeeze back to [Batch, Hidden]
        return attn_output.squeeze(1), attn_weights

# ==========================================
# 3. THE THREE MODELS
# ==========================================

class AudioOnlyModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        # Classifier takes 128 dim audio features
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, audio, video=None): 
        # Note: 'video' argument is ignored here, keeping API consistent
        features = self.audio_encoder(audio)
        logits = self.classifier(features)
        return logits

class LateFusionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()
        
        # Fusion: Concatenate Audio (128) + Video Mean (512) = 640
        self.classifier = nn.Sequential(
            nn.Linear(128 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, video):
        # 1. Audio Features: [Batch, 128]
        a_feat = self.audio_encoder(audio)
        
        # 2. Video Features: [Batch, Frames, 512]
        v_feat_seq = self.video_encoder(video)
        
        # 3. Simple Temporal Pooling (Mean over frames)
        # We just average the 5 frames to get one vector: [Batch, 512]
        v_feat = torch.mean(v_feat_seq, dim=1)
        
        # 4. Concatenate
        combined = torch.cat((a_feat, v_feat), dim=1)
        
        # 5. Classify
        logits = self.classifier(combined)
        return logits

class CrossAttentionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()
        
        # Attention Module
        # We project everything to 256 dimensions internally
        self.attention = CrossModalAttention(audio_dim=128, video_dim=512, hidden_dim=256)
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, audio, video):
        # 1. Extract Features
        a_feat = self.audio_encoder(audio)       # [Batch, 128]
        v_feat_seq = self.video_encoder(video)   # [Batch, Frames, 512]
        
        # 2. Apply Cross Attention
        # The Audio 'queries' the Video frames
        # context: [Batch, 256]
        context, weights = self.attention(a_feat, v_feat_seq)
        
        # 3. Classify
        logits = self.classifier(context)
        
        # Return logits AND weights (useful for visualization later)
        return logits, weights