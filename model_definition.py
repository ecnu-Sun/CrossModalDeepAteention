# model_definition.py

import torch
import torch.nn as nn
from transformers import CLIPModel
import torch.nn.functional as F
from cross_modal_attention import CrossModalTransformer


class MultimodalCrossAttentionClassifier(nn.Module):
    """
    这是本次实验中效果最好的模型，同时也比较吃算力，模型大小约1000M
    模型架构：
    1.使用预训练CLIP的文本/图像编码器获取序列特征（CLIP的隐藏层状态(BatchSize,Len,dim)包含了文本/图像每个token/patch的信息），
    2.再利用自定义的CrossModalTransformer进行图像和文本信息的深度融合
    （和浅融合对应，如果只是对文本/图像的最终的在同一空间中的512维嵌入向量进行互注意力融合，那么这里就叫做浅融合），
    3.最后在融合后的文本序列/图像序列上进行池化，拼接后用MLP情感三分类。
    """

    def __init__(self, pretrained_model_name="./models/clip-vit-base-patch32", hidden_dim=768, num_labels=3):
        super().__init__()
        # CLIP基础模型
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)
        self.text_projection = nn.Linear(512, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # 跨模态Transformer
        self.cross_modal_transformer = CrossModalTransformer(
            hidden_dim=self.hidden_dim,
            num_heads=32,
            num_layers=8
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )


    def forward(self, input_ids, attention_mask, pixel_values):
        # CLIPModel内部： text_model => text_outputs.last_hidden_state (B,L_t,dim)
        #                vision_model => vision_outputs.last_hidden_state (B,L_i,dim)
        MAX_SEQ_LENGTH = 77

        # CLIP 模型最大支持77长度，所以要截断
        input_ids = input_ids[:, :MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]

        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        text_feats = text_outputs.last_hidden_state  # (B, L_t, text_hidden_size)
        image_feats = vision_outputs.last_hidden_state  # (B, L_i, vision_hidden_size)
        print(text_feats.shape)
        print(image_feats.shape)
        text_feats = self.text_projection(text_feats)
        print(text_feats.shape)
        text_feats, image_feats = self.cross_modal_transformer(text_feats, image_feats)

        # 池化
        text_cls = text_feats[:, 0, :]  # (B, dim)
        image_pooled = torch.mean(image_feats, dim=1)  # (B, dim)

        # 拼接
        combined_feats = torch.cat((text_cls, image_pooled), dim=-1)  # (B, 2*dim)
        logits = self.classifier(combined_feats)  # (B, num_labels)

        return logits


class MultimodalClipClassifier(nn.Module):
    """
    使用预训练CLIP的最终文本特征和图像特征（已经对齐到同一个特征空间，都是512维）,
    直接拼接后通过MLP进行分类。
    """

    def __init__(self, pretrained_model_name="./models/clip-vit-base-patch32", num_labels=3):
        super().__init__()

        # 1) 加载预训练的 CLIP 模型
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)

        # 2) 定义分类头
        #    CLIP text/image feature 默认维度一般是 512，因此拼接后是 1024。
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):

        MAX_SEQ_LENGTH = 77
        input_ids = input_ids[:, :MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]
        text_embeds = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
        combined_feats = torch.cat([text_embeds, image_embeds], dim=-1)
        logits = self.classifier(combined_feats)

        return logits


class MultimodalClipClassifierWithAttention(nn.Module):
    """
    使用CLIP得到文本特征和图像最终特征，并使用互注意力进行浅融合。
    """

    def __init__(self, pretrained_model_name="./models/clip-vit-base-patch32",
                 num_labels=3, num_heads=4):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)

        self.attention_block = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=num_heads,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):

        MAX_SEQ_LENGTH = 77
        input_ids = input_ids[:, :MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]

        text_embeds = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
        tokens = torch.cat([text_embeds.unsqueeze(1),
                            image_embeds.unsqueeze(1)], dim=1)

        #    注意力：让文本 token 与图像 token 互相“看”对方
        #    fused_tokens 形状仍是 (batch_size, 2, 512)
        fused_tokens, _ = self.attention_block(tokens, tokens, tokens)

        pooled = fused_tokens.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


class MultimodalClipClassifierWithInfoNCE(nn.Module):
    """
    在原来的“text_embeds + image_embeds拼接做分类”的基础上，
    额外加入对比损失(InfoNCE) 以保持CLIP的特性，但是本模型不做注意力机制
    """

    def __init__(self, pretrained_model_name="./models/clip-vit-base-patch32",
                 num_labels=3,
                 contrastive_lambda=0.1,
                 contrastive_temp=0.07):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temp = contrastive_temp

    def forward(self,
                input_ids,
                attention_mask,
                pixel_values,
                labels=None):
        MAX_SEQ_LENGTH = 77
        input_ids = input_ids[:, :MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]

        text_embeds = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)

        combined_feats = torch.cat([text_embeds, image_embeds], dim=-1)
        logits = self.classifier(combined_feats)

        if labels is None:
            return logits

        loss_ce = F.cross_entropy(logits, labels)

        loss_contrast = self._compute_contrastive_loss(text_embeds,
                                                       image_embeds,
                                                       temp=self.contrastive_temp)

        loss_total = loss_ce + self.contrastive_lambda * loss_contrast

        return loss_total, logits

    def _compute_contrastive_loss(self, text_embeds, image_embeds, temp=0.07):
        """
        计算一批 (text_embeds, image_embeds) 的 InfoNCE 对比损失。
        做法与 CLIP 的原理类似，对称地约束 text->image, image->text。
        """
        text_normed = F.normalize(text_embeds, dim=-1)
        image_normed = F.normalize(image_embeds, dim=-1)

        logits_per_text = text_normed @ image_normed.transpose(-1, -2)  # [B, B]
        logits_per_image = image_normed @ text_normed.transpose(-1, -2)  # [B, B]

        logits_per_text = logits_per_text / temp
        logits_per_image = logits_per_image / temp

        batch_size = text_embeds.size(0)
        labels = torch.arange(batch_size, device=text_embeds.device)

        loss_text = F.cross_entropy(logits_per_text, labels)
        loss_image = F.cross_entropy(logits_per_image, labels)

        contrast_loss = (loss_text + loss_image) / 2.0
        return contrast_loss


class MultimodalClipClassifierWithWeight(nn.Module):
    """
    在原来的“text_embeds + image_embeds拼接做分类”的基础上，
    先对text_embeds和image_embeds加可学习权重，再分类
    """

    def __init__(self, pretrained_model_name="./models/clip-vit-base-patch32", num_labels=3):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)


        self.text_alpha = nn.Parameter(torch.tensor(1.0))
        self.image_alpha = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        MAX_SEQ_LENGTH = 77
        input_ids = input_ids[:, :MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:, :MAX_SEQ_LENGTH]
        text_embeds = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)

        # 对文本和图像特征进行可学习的缩放
        text_embeds_scaled = self.text_alpha * text_embeds
        image_embeds_scaled = self.image_alpha * image_embeds

        combined_feats = torch.cat([text_embeds_scaled, image_embeds_scaled], dim=-1)

        logits = self.classifier(combined_feats)

        return logits
