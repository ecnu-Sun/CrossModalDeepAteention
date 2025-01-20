# cross_modal_attention.py

import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """
    单层双向跨模态注意力:
    先让X (query) attend到 Y (key, value)，再让 Y attend 到 X。
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.x_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.y_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.x_ln1 = nn.LayerNorm(hidden_dim)
        self.y_ln1 = nn.LayerNorm(hidden_dim)
        self.x_ln2 = nn.LayerNorm(hidden_dim)
        self.y_ln2 = nn.LayerNorm(hidden_dim)

        self.x_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim),
        )
        self.y_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim),
        )

    def forward(self, x, y):
        # x -> y
        x_res = x
        x2, _ = self.x_attn(query=x, key=y, value=y)
        x = self.x_ln1(x_res + x2)
        x_res = x
        x2 = self.x_ffn(x)
        x = self.x_ln2(x_res + x2)

        # y -> x
        y_res = y
        y2, _ = self.y_attn(query=y, key=x, value=x)
        y = self.y_ln1(y_res + y2)
        y_res = y
        y2 = self.y_ffn(y)
        y = self.y_ln2(y_res + y2)

        return x, y

class CrossModalTransformer(nn.Module):
    """
    多层跨模态Transformer，每层执行双向CrossAttention。
    """
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, text_feats, image_feats):
        """
        text_feats: (batch_size, L_text, hidden_dim)
        image_feats: (batch_size, L_image, hidden_dim)
        """
        x, y = text_feats, image_feats
        for layer in self.layers:
            x, y = layer(x, y)
        return x, y
