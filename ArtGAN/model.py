import torch
import torch.nn as nn
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Attention mechanism over the RNN sequence to focus on the most informative spatial locations."""
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_output):
        # rnn_output: (B, SeqLen, hidden_dim)
        attn_weights = self.attention(rnn_output)  # (B, SeqLen, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(rnn_output * attn_weights, dim=1)  # (B, hidden_dim)
        return context


class ConvRNNHybrid(nn.Module):
    def __init__(self, num_styles=27, num_artists=23, num_genres=10,
                 hidden_dim=512, num_rnn_layers=2, dropout=0.3, freeze_backbone=True):
        super(ConvRNNHybrid, self).__init__()

        # --- CNN Backbone (ResNet50) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_out_channels = 2048

        # Freeze early layers of the backbone to prevent overfitting
        if freeze_backbone:
            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the last two ResNet blocks (layer3 and layer4) for fine-tuning
            for name, param in self.backbone.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True

        # --- Batch Norm after CNN ---
        self.bn = nn.BatchNorm1d(self.cnn_out_channels)

        # --- RNN Sequential Processor ---
        self.rnn = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_rnn_layers > 1 else 0
        )

        rnn_out_dim = hidden_dim * 2  # Bidirectional

        # --- Attention ---
        self.attention = SpatialAttention(rnn_out_dim)

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout)

        # --- Classification Heads (with hidden layer for better representation) ---
        self.style_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_styles)
        )
        self.artist_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_artists)
        )
        self.genre_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_genres)
        )

    def forward(self, x):
        # 1. CNN Feature Extraction
        features = self.backbone(x)  # (B, 2048, 7, 7)

        # 2. Reshape for RNN
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)  # (B, 2048, 49)

        # Apply batch norm across the channel dimension
        features = self.bn(features)

        # Permute to (B, SeqLen, Features)
        sequence = features.permute(0, 2, 1)  # (B, 49, 2048)

        # 3. RNN Processing
        rnn_out, _ = self.rnn(sequence)  # (B, 49, 2*hidden_dim)

        # 4. Attention-based aggregation (instead of simple mean)
        context_vector = self.attention(rnn_out)  # (B, 2*hidden_dim)
        context_vector = self.dropout(context_vector)

        # 5. Multi-label Prediction
        style_logits = self.style_head(context_vector)
        artist_logits = self.artist_head(context_vector)
        genre_logits = self.genre_head(context_vector)

        return {
            'style': style_logits,
            'artist': artist_logits,
            'genre': genre_logits
        }


if __name__ == '__main__':
    print("Testing ConvRNNHybrid with dummy input...")
    model = ConvRNNHybrid()
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)

    print(f"Style logits shape: {outputs['style'].shape} (Expected: 2, 27)")
    print(f"Artist logits shape: {outputs['artist'].shape} (Expected: 2, 23)")
    print(f"Genre logits shape: {outputs['genre'].shape} (Expected: 2, 10)")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print("Model test successful.")
